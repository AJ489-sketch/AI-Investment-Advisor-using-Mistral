# app.py (corrected: robust sales/price merge, timezone-safe)
import json
import io
import math
from datetime import datetime, timezone

import yfinance as yf
import feedparser
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

# 🔹 NEW IMPORTS FOR FINE-TUNED MISTRAL
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# ---------- CONFIG ----------
BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
LORA_PATH = "./mistral-finetuned"
CHART_DPI = 120
# ----------------------------

# ---------- LOAD FINE-TUNED MISTRAL (ONCE) ----------
print("🔹 Loading base Mistral model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map="auto",
    load_in_4bit=True
)

print("🔹 Attaching LoRA adapters...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
tokenizer.pad_token = tokenizer.eos_token

model.eval()
print("✅ Fine-tuned Mistral loaded successfully")

# ---------- Helpers ----------
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

def placeholder_image(text="No data", size=(900,400), bgcolor=(250,250,250)):
    img = Image.new("RGB", size, bgcolor)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    w, h = draw.textsize(text, font=font)
    draw.text(((size[0]-w)/2, (size[1]-h)/2), text, fill="black", font=font)
    return img

# ---------- LLM CALL (FINE-TUNED MISTRAL) ----------
def call_llm(prompt: str):
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=True,
                temperature=0.3,
                top_p=0.9
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    except Exception as e:
        return f"LLM error: {e}"

# ---------- Fundamentals & Industry ----------
def get_fundamentals_with_industry(ticker: str):
    tk = yf.Ticker(ticker)
    info = tk.info or {}
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")

    fundamentals = {
        "shortName": info.get("shortName"),
        "sector": sector,
        "industry": industry,
        "marketCap": info.get("marketCap"),
        "trailingPE": safe_float(info.get("trailingPE")),
        "forwardPE": safe_float(info.get("forwardPE")),
        "pegRatio": safe_float(info.get("pegRatio")),
        "priceToBook": safe_float(info.get("priceToBook")),
        "returnOnEquity": safe_float(info.get("returnOnEquity")),
        "returnOnAssets": safe_float(info.get("returnOnAssets")),
        "profitMargins": safe_float(info.get("profitMargins")),
        "debtToEquity": safe_float(info.get("debtToEquity")),
        "revenueGrowth": safe_float(info.get("revenueGrowth")),
        "earningsQuarterlyGrowth": safe_float(info.get("earningsQuarterlyGrowth")),
        "beta": safe_float(info.get("beta")),
    }

    industry_avg = {
        "trailingPE": 21.4,
        "forwardPE": 19.8,
        "pegRatio": 1.6,
        "priceToBook": 4.0,
        "returnOnEquity": 0.14,
        "returnOnAssets": 0.08,
        "profitMargins": 0.12,
        "debtToEquity": 0.8,
        "revenueGrowth": 0.08,
        "earningsQuarterlyGrowth": 0.09,
        "beta": 1.0,
    }

    rows = []
    for k, v in fundamentals.items():
        rows.append({
            "Metric": k,
            "Company": v if v is not None else "N/A",
            "Industry Avg": industry_avg.get(k, "N/A")
        })

    return pd.DataFrame(rows), sector, industry

# ---------- News ----------
def get_news(ticker: str, max_items: int = 6):
    url = f"https://news.google.com/rss/search?q={ticker}+stock"
    feed = feedparser.parse(url)
    items = []

    for i, entry in enumerate(feed.entries[:max_items]):
        items.append({
            "title": entry.title,
            "link": getattr(entry, "link", ""),
            "published": getattr(entry, "published", ""),
            "weight": round(max(0.05, 1 - i * 0.15), 3)
        })

    return pd.DataFrame(items)

# ---------- Sales & Correlation ----------
def fetch_sales_and_correlation(ticker: str):
    tk = yf.Ticker(ticker)
    fin = tk.quarterly_financials
    hist = tk.history(period="2y", auto_adjust=True)

    if fin is None or fin.empty:
        return pd.DataFrame(), None, pd.DataFrame()

    revenue_row = next((i for i in fin.index if "revenue" in str(i).lower()), None)
    if revenue_row is None:
        return pd.DataFrame(), None, pd.DataFrame()

    rev_df = fin.loc[revenue_row].reset_index()
    rev_df.columns = ["Date", "Revenue"]
    rev_df["Date"] = pd.to_datetime(rev_df["Date"])

    close_df = hist[["Close"]].reset_index()
    close_df["Date"] = pd.to_datetime(close_df["Date"])

    merged = pd.merge_asof(
        rev_df.sort_values("Date"),
        close_df.sort_values("Date"),
        on="Date",
        direction="backward"
    )

    corr = None
    try:
        corr = float(
            np.corrcoef(
                merged["Revenue"].pct_change().dropna(),
                merged["Close"].pct_change().dropna()
            )[0, 1]
        )
    except Exception:
        pass

    return merged, corr, rev_df

def plot_sales_chart(merged_df: pd.DataFrame, ticker: str):
    if merged_df.empty:
        return placeholder_image("No sales data available")

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.bar(merged_df["Date"], merged_df["Revenue"]/1e9, alpha=0.7)
    ax1.set_ylabel("Revenue (Billion)")
    ax1.set_title(f"{ticker} — Revenue vs Price")

    ax2 = ax1.twinx()
    ax2.plot(merged_df["Date"], merged_df["Close"], marker="o")
    ax2.set_ylabel("Close Price")

    return fig_to_pil(fig)

# ---------- Main Analyze ----------
def analyze_stock(ticker: str, question: str, scenario_pct: float):
    ticker = ticker.upper()

    fundamentals_df, sector, industry = get_fundamentals_with_industry(ticker)
    news_df = get_news(ticker)
    merged_sales, corr, _ = fetch_sales_and_correlation(ticker)
    sales_chart = plot_sales_chart(merged_sales, ticker)

    prompt = f"""
You are a professional equity research analyst.

Stock: {ticker}
Sector: {sector}
Industry: {industry}

Fundamentals:
{fundamentals_df.to_string(index=False)}

News:
{news_df.to_string(index=False)}

Sales-Price Correlation: {corr}

User Question:
{question}

Provide a concise BUY / HOLD / SELL recommendation with numeric justification.
"""

    insight = call_llm(prompt)

    sales_table = (
        merged_sales[["Date", "Revenue", "Close"]]
        .rename(columns={"Close": "Price"})
        if not merged_sales.empty else pd.DataFrame()
    )

    return insight, fundamentals_df, news_df, sales_table, sales_chart, ""

# ---------- Gradio UI ----------
with gr.Blocks(title="AI Stock Advisor — Fine-tuned Mistral") as demo:
    gr.Markdown("# 📈 AI Stock Advisor (Fine-Tuned Mistral)")

    with gr.Row():
        with gr.Column(scale=3):
            ticker_input = gr.Textbox(label="Stock Ticker", value="AAPL")
            question_input = gr.Textbox(label="Your Question", value="Provide a concise analyst recommendation.")
            scenario_input = gr.Number(label="What-if Revenue Growth (%)", value=0)
            run_btn = gr.Button("Run Analysis", variant="primary")

        with gr.Column(scale=7):
            result_md = gr.Markdown()
            fundamentals_table = gr.Dataframe()
            news_table = gr.Dataframe()
            sales_table = gr.Dataframe()
            sales_chart = gr.Image()

    run_btn.click(
        fn=analyze_stock,
        inputs=[ticker_input, question_input, scenario_input],
        outputs=[result_md, fundamentals_table, news_table, sales_table, sales_chart, gr.Textbox()]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
