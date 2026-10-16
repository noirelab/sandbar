"""Agent that answers questions about exposed sandbars on the Araguaia from Sentinel-2.

An LLM (DeepSeek, OpenAI-compatible API) plans the measurements, calls the tools in
tools.py and reports a short answer. The key comes from the DEEPSEEK_API_KEY env var or,
if unset, from ~/.config/deepseek/key (chmod 600, never committed).

Usage: uv run agent.py "question" [model] [effort]   (effort: low | medium | high | off)
"""
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

from openai import OpenAI

import tools

# USD per million tokens at peak hours (cache hit, cache miss, output); off-peak is half.
PRICES = {"deepseek-flash": (0.006, 0.3, 1.2), "deepseek-v4-pro": (0.044, 1.32, 3.96)}
MAX_STEPS = 30
PROMPT = "v2"  # tags eval results, so prompt revisions can be compared

SYSTEM = """You answer questions about exposed sand on the Araguaia River, Brazil, measured from \
Sentinel-2 satellite imagery. Exposed sand grows as the river drops in the dry season (June to \
October), so it is a proxy for drought.

Use the tools for every number; never estimate from memory. A measurement covers one reach \
(a ~20 x 18 km box) and one calendar month, using the clearest scene of that month that covers the reach. \
`unet` is the trained segmentation model and the default; `baseline` is the hand-written spectral \
rule. Use `baseline` only when the question asks for the rule or a comparison with it.

Only the reaches from `list_reaches` can be measured, and each covers only its own box. A town \
that names a reach (for example Aruanã for `aruana`) is that reach. Any other place, even a town \
nearby, on another river, or with a similar name, has no data: never substitute the nearest reach.

If a tool returns an error, retry the same call at most once. If the measurement still fails, \
the answer is "no data"; never switch to another method unless the user asked for it.

For averages, differences and comparisons, write the numbers and the calculation out before \
deciding.

When you know the answer, call `answer` once with the bare value: a number in the unit the \
question implies (km2, percent), a reach name, or a year. If the data cannot answer the question \
(unknown reach, no clear scene), call `answer` with "no data". Then give the user a brief reply \
in their language, citing the scene dates you used."""


def _tool(name, description, properties=None, required=()):
    schema = {"type": "object", "properties": properties or {}, "required": list(required), "additionalProperties": False}
    return {"type": "function", "function": {"name": name, "description": description, "parameters": schema}}


TOOLS = [
    _tool("list_reaches", "List the river reaches that can be measured, ordered from upstream (south) to downstream (north)."),
    _tool(
        "measure",
        "Measure exposed sand and water area (km2) in one reach for one calendar month. "
        "Downloads the scene on first use, which can take about two minutes.",
        {
            "reach": {"type": "string", "description": "Reach name from list_reaches."},
            "year": {"type": "integer", "description": "Four-digit year, 2017 or later."},
            "month": {"type": "integer", "description": "Month number, 1 to 12."},
            "method": {"type": "string", "enum": ["unet", "baseline"], "description": "unet (trained model, default) or baseline (spectral rule)."},
        },
        ["reach", "year", "month"],
    ),
    _tool(
        "answer",
        "Record the final short answer before replying to the user.",
        {"value": {"type": "string", "description": 'Bare value: a number, a reach name, a year, or "no data".'}},
        ["value"],
    ),
]

_client = None


def client():
    global _client
    if _client is None:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            path = Path("~/.config/deepseek/key").expanduser()
            if not path.exists():
                sys.exit("No DeepSeek key: set DEEPSEEK_API_KEY or create ~/.config/deepseek/key")
            key = path.read_text().strip()
        _client = OpenAI(base_url="https://api.deepseek.com", api_key=key)
    return _client


def call_tool(name, args, trace, answers, broken={}):
    if name == "list_reaches":
        return tools.reaches()
    if name == "answer":
        answers.append(str(args.get("value")))
        return "recorded"
    if name == "measure":
        if args.get("method", "unet") in broken:  # eval-only fault injection
            result = {"error": broken[args.get("method", "unet")]}
            trace.append({"tool": "measure", "input": args, "output": result})
            return result
        try:
            result = tools.measure(args["reach"], int(args["year"]), int(args["month"]), args.get("method", "unet"))
        except (KeyError, ValueError) as e:
            result = {"error": str(e)}
        trace.append({"tool": "measure", "input": args, "output": result})
        return result
    return {"error": f"unknown tool {name}"}


def run(question, model="deepseek-flash", effort="high", broken={}):
    """Answer one question; returns the answer, the tool trace, cost and latency."""
    thinking = {"thinking": {"type": "disabled"}} if effort == "off" else {"thinking": {"type": "enabled"}}
    effort_arg = {} if effort == "off" else {"reasoning_effort": effort}
    messages = [{"role": "system", "content": f"{SYSTEM}\n\nToday's date is {date.today()}."}, {"role": "user", "content": question}]
    trace, answers = [], []
    hit = miss = out = calls = 0
    start = time.time()
    for _ in range(MAX_STEPS):
        resp = client().chat.completions.create(model=model, messages=messages, tools=TOOLS, extra_body=thinking, **effort_arg)
        u = resp.usage
        hit += getattr(u, "prompt_cache_hit_tokens", 0) or 0
        miss += getattr(u, "prompt_cache_miss_tokens", u.prompt_tokens) or 0
        out += u.completion_tokens
        msg = resp.choices[0].message
        # With tools, DeepSeek requires reasoning_content to be sent back on every later request.
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            break
        for tc in msg.tool_calls:
            calls += 1
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                result = {"error": "arguments were not valid JSON"}
            else:
                result = call_tool(tc.function.name, args, trace, answers, broken)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    p_hit, p_miss, p_out = PRICES.get(model, (0, 0, 0))
    return {
        "answer": answers[-1] if answers else None,
        "reply": msg.content or "",
        "finish_reason": resp.choices[0].finish_reason,
        "tool_calls": calls,
        "measurements": trace,
        "tokens_in": hit + miss,
        "tokens_out": out,
        "cost_usd": round((hit * p_hit + miss * p_miss + out * p_out) / 1e6, 5),
        "seconds": round(time.time() - start, 1),
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    model = args[1] if len(args) > 1 else "deepseek-flash"
    effort = args[2] if len(args) > 2 else "high"
    result = run(args[0], model, effort)
    print(result["reply"])
    print(f"\nanswer={result['answer']}  calls={result['tool_calls']}  ${result['cost_usd']}  {result['seconds']}s")
