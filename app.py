# app.py — Clean UI version (distilgpt2 placeholder)

import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import yfinance as yf
import feedparser
import gradio as gr
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- CONFIG ----------------
MODEL_NAME = "distilgpt2"
CHART_DPI = 120
# --------------------------------------


# ---------------- LOAD MODEL ----------------
print("🔹 Loading lightweight model (distilgpt2)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token
model.eval()

print("✅ distilgpt2 loaded successfully")
# -------------------------------------------


# ---------------- HELPERS ----------------
def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=CHART_DPI, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    plt.close(fig)
    return img


def placeholder_image(text="No data", size=(900, 400)):
    img = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    w, h = draw.textsize(text, font=font)
    draw.text(((size[0]-w)/2, (size[1]-h)/2), text, fill="black", font=font)
    return img
# -----------------------------------------


# ---------------- LLM CALL ----------------
def call_llm(ticker: str, question: str):
    """
    distilgpt2 is NOT instruction-tuned.
    So we keep the prompt very short and focused.
    """

    prompt = f"{ticker} stock analysis: {question}\nAnswer:"

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt echo
        if "Answer:" in text:
            text = text.split("Answer:")[-1]

        return text.strip()

    except Exception as e:
        return f"LLM error: {e}"
# -----------------------------------------


# ---------------- FUNDAMENTALS ----------------
def get_fundamentals(ticker: str):
    tk = yf.Ticker(ticker)
    info = tk.info or {}

    data = {
        "Metric": [
            "Market Capitalization",
            "Trailing PE",
            "Forward PE",
            "Price to Book",
            "Return on Equity",
            "Return on Assets",
            "Profit Margin",
            "Debt to Equity",
            "Revenue Growth",
            "Beta",
        ],
        "Value": [
            info.get("marketCap"),
            safe_float(info.get("trailingPE")),
            safe_float(info.get("forwardPE")),
            safe_float(info.get("priceToBook")),
            safe_float(info.get("returnOnEquity")),
            safe_float(info.get("returnOnAssets")),
            safe_float(info.get("profitMargins")),
            safe_float(info.get("debtToEquity")),
            safe_float(info.get("revenueGrowth")),
            safe_float(info.get("beta")),
        ],
    }

    return pd.DataFrame(data)
# ----------------------------------------------


# ---------------- NEWS ----------------
def get_news(ticker: str, max_items=5):
    url = f"https://news.google.com/rss/search?q={ticker}+stock"
    feed = feedparser.parse(url)

    items = []
    for entry in feed.entries[:max_items]:
        items.append({
            "Headline": entry.title,
            "Published": getattr(entry, "published", "")
        })

    return pd.DataFrame(items)
# -------------------------------------


# ---------------- SALES CHART ----------------
def plot_revenue_chart(ticker: str):
    tk = yf.Ticker(ticker)
    fin = tk.quarterly_financials

    if fin is None or fin.empty:
        return placeholder_image("No quarterly revenue data available")

    revenue_row = next((i for i in fin.index if "revenue" in str(i).lower()), None)
    if revenue_row is None:
        return placeholder_image("No revenue data available")

    df = fin.loc[revenue_row].reset_index()
    df.columns = ["Date", "Revenue"]
    df["Date"] = pd.to_datetime(df["Date"])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df["Date"], df["Revenue"] / 1e9)
    ax.set_title(f"{ticker} — Quarterly Revenue")
    ax.set_ylabel("Revenue (Billion USD)")
    ax.set_xlabel("Quarter")

    return fig_to_pil(fig)
# ---------------------------------------------


# ---------------- MAIN ANALYSIS ----------------
def analyze_stock(ticker: str, question: str):
    ticker = ticker.upper()

    ai_analysis = call_llm(ticker, question)
    fundamentals = get_fundamentals(ticker)
    news = get_news(ticker)
    chart = plot_revenue_chart(ticker)

    return ai_analysis, fundamentals, news, chart
# -----------------------------------------------


# ---------------- GRADIO UI ----------------
with gr.Blocks(title="AI Stock Advisor (Demo)") as demo:
    gr.Markdown("## 📈 AI Stock Advisor (Demo – distilgpt2)")
    gr.Markdown(
        "⚠️ *This demo uses a lightweight language model for UI testing. "
        "Final version will use a fine-tuned Mistral model.*"
    )

    with gr.Row():
        with gr.Column(scale=3):
            ticker_input = gr.Textbox(label="Stock Ticker", value="AAPL")
            question_input = gr.Textbox(
                label="Your Question",
                value="Give a short analyst-style comment"
            )
            run_btn = gr.Button("Run Analysis", variant="primary")

        with gr.Column(scale=7):
            gr.Markdown("### 🧠 AI Analysis")
            ai_output = gr.Markdown()

            gr.Markdown("### 📊 Key Fundamentals")
            fundamentals_table = gr.Dataframe()

            gr.Markdown("### 📰 Recent News")
            news_table = gr.Dataframe()

            gr.Markdown("### 📉 Quarterly Revenue Trend")
            revenue_chart = gr.Image()

    run_btn.click(
        fn=analyze_stock,
        inputs=[ticker_input, question_input],
        outputs=[ai_output, fundamentals_table, news_table, revenue_chart]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
