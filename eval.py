"""Eval for the agent: questions whose answers come from the same deterministic tools.

The ground truth tests planning, tool use and arithmetic, not segmentation accuracy:
both the truth and the agent read tools.measure, so a perfect agent scores 100%.

Usage:
  uv run eval.py build                  writes eval/questions.jsonl from the scene cache
  uv run eval.py run MODEL [EFFORT] [N] runs the first N questions, writes eval/results_*.jsonl
"""
import json
import random
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median

import tools

OUT = Path("eval")
YEARS = (2023, 2024, 2025, 2026)
MONTHS = {7: "julho", 9: "setembro"}
UNKNOWN = ["Marabá", "Imperatriz", "Porto Nacional", "Belém"]


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[\s_]+", "-", s.strip().lower())


def sand(r, y, m, method):
    try:
        return tools.measure(r, y, m, method)["sand_km2"]
    except ValueError:
        return None


def build(seed=0):
    method = "unet" if Path("unet.pt").exists() else "baseline"
    hint = "" if method == "unet" else " Use a regra espectral."
    rs = [r["reach"] for r in tools.reaches()]
    have = lambda y, m: {r: v for r in rs if (v := sand(r, y, m, method)) is not None}
    rnd = random.Random(seed)
    qs = []

    def add(kind, text, truth, tol=0.01):
        qs.append({"id": f"{kind}-{len(qs)}", "kind": kind, "question": text + hint, "truth": truth, "abs_tol": tol, "method": method})

    for _ in range(10):
        r, y, m = rnd.choice(rs), rnd.choice(YEARS), rnd.choice(list(MONTHS))
        if (v := sand(r, y, m, method)) is not None:
            add("single", f"Quantos km² de areia exposta havia no trecho {r} em {MONTHS[m]} de {y}?", v)
    for _ in range(10):
        r, (y1, y2) = rnd.choice(rs), sorted(rnd.sample(YEARS, 2))
        a, b = sand(r, y1, 9, method), sand(r, y2, 9, method)
        if a is not None and b is not None:
            add("diff", f"Em {r}, quanto a areia exposta de setembro variou de {y1} para {y2}, em km² ({y2} menos {y1})?", round(b - a, 3))
    for _ in range(10):
        r, y = rnd.choice(rs), rnd.choice(YEARS)
        a, b = sand(r, y, 7, method), sand(r, y, 9, method)
        if a and b is not None:
            add("season_pct", f"Em {y}, em quantos por cento a areia exposta em {r} cresceu de julho para setembro?", round((b - a) / a * 100, 1), tol=1.0)
    for y in YEARS:
        if len(v := have(y, 9)) == len(rs):
            add("rank", f"Em setembro de {y}, qual trecho do Araguaia tinha a maior área de areia exposta?", max(v, key=v.get))
    for r in rnd.sample(rs, 6):
        v = {y: s for y in YEARS if (s := sand(r, y, 9, method)) is not None}
        if len(v) == len(YEARS):
            add("max_year", f"Entre 2023 e 2026, em que ano o trecho {r} teve mais areia exposta em setembro?", str(max(v, key=v.get)))
    for y in rnd.sample(YEARS, 2):
        if len(v := have(y, 9)) == len(rs):
            add("total", f"Somando todos os trechos medidos, quantos km² de areia estavam expostos em setembro de {y}?", round(sum(v.values()), 3), tol=0.05)
    if method == "unet":
        for _ in range(6):
            r, y, m = rnd.choice(rs), rnd.choice(YEARS), rnd.choice(list(MONTHS))
            u, b = sand(r, y, m, "unet"), sand(r, y, m, "baseline")
            if u is not None and b is not None:
                add("method_gap", f"Em {r}, {MONTHS[m]} de {y}, quantos km² de areia a U-Net detecta a mais que a regra espectral?", round(u - b, 3))
    for place in UNKNOWN:
        add("no_data", f"Quanta areia exposta havia em {place} em setembro de 2024?", "no data")

    OUT.mkdir(exist_ok=True)
    (OUT / "questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs))
    print(f"{len(qs)} questions, method {method}")


def grade(q, answer):
    """correct / wrong / no_answer. Numbers pass within 1% or the case's absolute tolerance."""
    if answer is None:
        return "no_answer"
    if not isinstance(q["truth"], (int, float)):
        return "correct" if norm(answer) == norm(q["truth"]) else "wrong"
    m = re.search(r"-?\d+(?:[.,]\d+)?", answer)
    if not m:
        return "wrong"
    value = float(m.group().replace(",", "."))
    return "correct" if abs(value - q["truth"]) <= max(abs(q["truth"]) * 0.01, q["abs_tol"]) else "wrong"


def run(model, effort="high", n=None):
    import openai

    import agent

    agent.client()  # fail fast without a key, before any question runs
    qs = [json.loads(line) for line in (OUT / "questions.jsonl").read_text().splitlines()][:n]

    def one(q):
        try:
            r = agent.run(q["question"], model, effort)
        except openai.APIError as e:  # infra failure, not a model failure
            return {**q, "grade": "error", "error": repr(e)}
        return {**q, **r, "grade": grade(q, r["answer"])}

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(one, qs))
    path = OUT / f"results_{model}_{effort}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in results))

    ok = [r for r in results if r["grade"] != "error"]
    count = lambda g: sum(r["grade"] == g for r in results)
    print(f"{model} effort={effort}  n={len(results)}  correct {count('correct') / max(len(ok), 1):.0%}  "
          f"wrong {count('wrong')}  no_answer {count('no_answer')}  errors {count('error')}")
    if ok:
        print(f"tool calls/question {sum(r['tool_calls'] for r in ok) / len(ok):.1f}  "
              f"cost/question ${sum(r['cost_usd'] for r in ok) / len(ok):.4f}  "
              f"median latency {median(r['seconds'] for r in ok):.0f}s  total ${sum(r['cost_usd'] for r in ok):.2f}")
    for kind in sorted({r["kind"] for r in ok}):
        rs = [r for r in ok if r["kind"] == kind]
        print(f"  {kind:11s} {sum(r['grade'] == 'correct' for r in rs)}/{len(rs)}")
    print(f"wrote {path}")


if __name__ == "__main__":
    assert grade({"truth": 2.141, "abs_tol": 0.01}, "2,14 km²") == "correct"
    assert grade({"truth": 2.141, "abs_tol": 0.01}, "2.5") == "wrong"
    assert grade({"truth": "sao-felix", "abs_tol": 0}, "São Félix") == "correct"
    assert grade({"truth": "no data", "abs_tol": 0}, None) == "no_answer"
    cmd, *args = sys.argv[1:]
    if cmd == "build":
        build()
    else:
        run(args[0], args[1] if len(args) > 1 else "high", int(args[2]) if len(args) > 2 else None)
