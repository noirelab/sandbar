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
# How people name the reaches, and towns near the river that are outside the study area.
ALIASES = {
    "barra-do-garcas": "Barra do Garças", "aruana": "Aruanã", "cocalinho": "Cocalinho", "luiz-alves": "Luiz Alves",
    "sao-felix": "São Félix do Araguaia", "santa-terezinha": "Santa Terezinha", "conceicao": "Conceição do Araguaia", "xambioa": "Xambioá",
}
OUTSIDE = ["Caseara", "Araguaína", "Nova Xavantina", "Palmas"]
MONTH_WORDS = {7: ["julho", "july", "jul", "7"], 9: ["setembro", "september", "set", "sep", "9"]}
TODAY_YEAR = 2026  # agent.py tells the model today's date; relative questions ("last year") use it


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

    # v2: cases the first run showed are easy to get silently wrong.
    for r in rnd.sample(rs, 8):
        y, m = rnd.choice(YEARS), rnd.choice(list(MONTHS))
        if (v := sand(r, y, m, method)) is not None:
            add("alias", f"Quanta areia exposta tinha em {ALIASES[r]} em {MONTHS[m]} de {y}, em km²?", v)
    for place in OUTSIDE:
        add("outside", f"Como estava a areia exposta do rio em {place} em setembro de 2025, em km²?", "no data")
    add("no_scene", "Quantos km² de areia exposta havia em xambioa em setembro de 2023?", "no data")
    for _ in range(5):
        r, y, m = rnd.choice(rs), rnd.choice(YEARS[:3]), rnd.choice([1, 2, 3, 12])
        v = sand(r, y, m, method)
        add("rainy", f"Quantos km² de areia exposta havia no trecho {r} em {m:02d}/{y}?", "no data" if v is None else v)
    for _ in range(6):
        r, y, m = rnd.choice(rs), rnd.choice(YEARS), rnd.choice(list(MONTHS))
        if sand(r, y, m, method) is not None:
            qs.append({"id": f"tool_failure-{len(qs)}", "kind": "tool_failure", "broken": {"unet": "unet service unavailable, try again later"}, "abs_tol": 0.01, "method": method, "truth": "no data",
                       "question": f"Segundo a U-Net, quantos km² de areia exposta havia em {r} em {MONTHS[m]} de {y}?"})
    for r in rnd.sample(rs, 3):
        y = rnd.choice(YEARS)
        if (w := sand(r, y, 9, method)) is not None:
            add("hectares", f"Quantos hectares de areia exposta havia em {r} em setembro de {y}?", round(w * 100, 1), tol=1.0)
    v7, v9 = have(2025, 7), have(2025, 9)
    growth = {r: (v9[r] - v7[r]) / v7[r] for r in rs if v7.get(r) and r in v9}
    add("multi_hop", "Qual trecho teve o maior crescimento percentual de areia exposta de julho para setembro de 2025?", max(growth, key=growth.get))
    add("multi_hop", "Em setembro de 2025, quantos km² separam o trecho com mais areia exposta do trecho com menos?", round(max(v9.values()) - min(v9.values()), 3))
    v25, v26 = have(2025, 9), have(2026, 9)
    add("multi_hop", "Quantos trechos tinham mais areia exposta em setembro de 2026 do que em setembro de 2025?", sum(v26[r] > v25[r] for r in rs), tol=0)
    add("multi_hop", "Qual foi a média de areia exposta por trecho em setembro de 2024, em km²?", round(sum(have(2024, 9).values()) / len(rs), 3))
    a7, a9 = sand("sao-felix", TODAY_YEAR - 1, 7, method), sand("sao-felix", TODAY_YEAR - 1, 9, method)
    add("vague", "Meu pai pesca em São Félix do Araguaia. A praia lá estava maior em julho ou em setembro do ano passado?", MONTH_WORDS[9 if a9 > a7 else 7])
    best = max(v25, key=v25.get)
    add("vague", "Vou acampar numa praia do Araguaia em setembro. Pelos dados do ano passado, qual trecho teve mais areia exposta?", [best, ALIASES[best]])
    past = [sand("conceicao", y, 9, method) for y in (2023, 2024, 2025)]
    now = sand("conceicao", TODAY_YEAR, 9, method)
    above = now > sum(past) / 3
    add("vague", "Em Conceição do Araguaia, a areia de setembro deste ano ficou acima ou abaixo da média de setembro de 2023 a 2025?",
        ["acima", "above", "maior", "higher"] if above else ["abaixo", "below", "menor", "lower"])
    a23, a24 = sand("aruana", 2023, 9, method), sand("aruana", 2024, 9, method)
    add("vague", "A seca de 2024 deixou mais praia em Aruanã que a de 2023? Me diga a diferença de areia em setembro, em km² (2024 menos 2023).", round(a24 - a23, 3))

    # Held out: written before the prompt was revised, to check that fixes generalize.
    for place in ["São Félix do Xingu", "Conceição do Tocantins", "Santa Terezinha de Goiás", "Araguaiana"]:
        add("outside_holdout", f"Quanta areia o rio expôs em {place} em julho de 2025, em km²?", "no data")
    for _ in range(4):
        r, y = rnd.choice(rs), rnd.choice(YEARS)
        if sand(r, y, 9, method) is not None:
            qs.append({"id": f"tool_failure_holdout-{len(qs)}", "kind": "tool_failure_holdout", "broken": {"unet": "TimeoutError: inference did not finish in 30 s"},
                       "abs_tol": 0.01, "method": method, "truth": "no data",
                       "question": f"Quero o número da U-Net: quantos km² de areia havia em {r} em setembro de {y}?"})

    OUT.mkdir(exist_ok=True)
    (OUT / "questions.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs))
    print(f"{len(qs)} questions, method {method}")


def grade(q, answer):
    """correct / wrong / no_answer. Numbers pass within 1% or the case's absolute tolerance."""
    if answer is None:
        return "no_answer"
    if not isinstance(q["truth"], (int, float)):
        truths = q["truth"] if isinstance(q["truth"], list) else [q["truth"]]
        return "correct" if norm(answer) in {norm(t) for t in truths} else "wrong"
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
            r = agent.run(q["question"], model, effort, broken=q.get("broken", {}))
        except openai.APIError as e:  # infra failure, not a model failure
            return {**q, "grade": "error", "error": repr(e)}
        return {**q, **r, "grade": grade(q, r["answer"])}

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(one, qs))
    path = OUT / f"results_{model}_{effort}_{agent.PROMPT}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in results))

    ok = [r for r in results if r["grade"] != "error"]
    count = lambda g: sum(r["grade"] == g for r in results)
    print(f"{model} effort={effort} prompt={agent.PROMPT}  n={len(results)}  correct {count('correct') / max(len(ok), 1):.0%}  "
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
    assert grade({"truth": ["setembro", "9"], "abs_tol": 0}, "Setembro") == "correct"
    cmd, *args = sys.argv[1:]
    if cmd == "build":
        build()
    else:
        run(args[0], args[1] if len(args) > 1 else "high", int(args[2]) if len(args) > 2 else None)
