"""Web demo: ask about drought on the Araguaia, see the answer, the scenes and the tool trace.

Reads demo/measurements.json and demo/img/ from export.py, so it needs no GPU.

Usage: uv run app.py
"""
import json
from pathlib import Path

import gradio as gr
import pandas as pd

import agent

DEMO = Path("demo")
MAX_QUESTIONS = 10  # per browser session, so a public demo cannot drain the API key

EXAMPLES = [
    "Qual trecho do Araguaia teve mais areia exposta em setembro de 2024?",
    "A seca de 2024 deixou mais praia em Aruanã que a de 2023?",
    "Meu pai pesca em São Félix do Araguaia. A praia lá estava maior em julho ou em setembro do ano passado?",
    "Quanta areia exposta havia em Imperatriz em setembro de 2024?",
    "Quanto a U-Net e a regra espectral discordam em Santa Terezinha em fevereiro de 2024?",
]


def timeline():
    rows = [r for r in json.loads((DEMO / "measurements.json").read_text()) if r["method"] == "unet" and r["month"] == 9]
    return pd.DataFrame(rows)[["reach", "year", "sand_km2"]].sort_values("year")


def ask(question, history, used):
    if used >= MAX_QUESTIONS:
        history = history + [{"role": "user", "content": question}, {"role": "assistant", "content": "Limite de perguntas desta sessão atingido."}]
        return history, [], None, "", used
    r = agent.run(question)
    images = []
    for m in r["measurements"]:
        a = m["input"]
        path = DEMO / "img" / f"{a.get('reach')}_{a.get('year')}-{int(a.get('month', 0)):02d}.jpg"
        if path.exists() and "error" not in m["output"]:
            caption = f"{m['output']['reach']} {m['output']['scene_date']}: {m['output']['sand_km2']} km² ({m['output']['method']})"
            images.append((str(path), caption))
    stats = f"**Resposta:** `{r['answer']}` · {r['tool_calls']} chamadas de ferramenta · ${r['cost_usd']:.4f} · {r['seconds']} s"
    history = history + [{"role": "user", "content": question}, {"role": "assistant", "content": r["reply"]}]
    return history, images, r["measurements"], stats, used + 1


with gr.Blocks(title="Seca no Araguaia") as demo:
    gr.Markdown(
        "# Seca no Araguaia\n"
        "Pergunte sobre a areia exposta em 8 trechos do rio Araguaia, de 2023 a 2026. "
        "Um LLM (DeepSeek) planeja as medições e chama uma U-Net que segmenta imagens Sentinel-2. "
        "Areia em amarelo, água em azul."
    )
    used = gr.State(0)
    with gr.Row():
        with gr.Column(scale=1):
            chat = gr.Chatbot(height=420)
            box = gr.Textbox(placeholder="Ex.: qual trecho secou mais em setembro de 2024?", show_label=False)
            gr.Examples(EXAMPLES, inputs=box)
            stats = gr.Markdown()
        with gr.Column(scale=1):
            gallery = gr.Gallery(label="Cenas usadas", columns=2, height=420)
            trace = gr.JSON(label="Chamadas de ferramenta")
    gr.LinePlot(timeline(), x="year", y="sand_km2", color="reach", title="Areia exposta em setembro (km², U-Net)")
    box.submit(ask, [box, chat, used], [chat, gallery, trace, stats, used]).then(lambda: "", None, box)

if __name__ == "__main__":
    demo.launch()
