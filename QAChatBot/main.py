# app.py
import os
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from dotenv import load_dotenv
import openai

load_dotenv()  # optional: load OPENAI_API_KEY from .env

# ---------------------------
# Configuration / UI
# ---------------------------
st.set_page_config(page_title="Multi-Model QA + Metrics", layout="wide")
st.title("Multi-Model Q&A Chatbot — Responses + Metrics")

st.markdown(
    """
This demo calls multiple OpenAI models in parallel for the same question, then shows:
- each model's answer  
- latency (time taken)  
- token usage (if returned)  
- pairwise semantic similarity between answers (via embeddings)
"""
)

# Read API key from environment
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.error("OpenAI API key not found in environment. Set OPENAI_API_KEY in your environment or .env file.")
    st.stop()

openai.api_key = OPENAI_KEY

with st.sidebar:
    st.header("Settings (no API key here)")
    models = st.multiselect(
        "Choose models to compare (examples):",
        options=[
            "gpt-4o", "gpt-4-turbo", "gpt-4",
            "gpt-3.5-turbo", "gpt-3.5-turbo-0613"
        ],
        default=["gpt-4o", "gpt-3.5-turbo"]
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    max_tokens = st.slider("Max tokens (response)", 50, 2000, 300, 50)
    concurrency = st.slider("Parallel requests (threads)", 1, min(8, max(1, len(models) or 1)), value=min(4, max(1, len(models) or 1)))
    st.markdown("---")
    st.markdown(
        "Tip: set `OPENAI_API_KEY` in your environment or `.env` file (this app reads it automatically)."
    )

# Main input
question = st.text_area("Question", height=120, placeholder="Ask something to compare how multiple models respond...")
run_button = st.button("Run")

if run_button:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()
    if not models:
        st.warning("Please select at least one model from the sidebar.")
        st.stop()

    # Prepare prompt/messages
    system_prompt = "You are a helpful assistant. Keep the answer concise and informative."
    user_message = question.strip()

    # Helper: call a single model and return metrics
    def call_model(model_name: str):
        start = time.perf_counter()
        try:
            # Use ChatCompletion to capture usage metrics when available
            resp = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency = time.perf_counter() - start

            # Extract answer text (concatenate if multiple choices)
            text = ""
            if "choices" in resp and len(resp.choices) > 0:
                text = "".join([c.message.get("content", "") for c in resp.choices])
            else:
                text = resp.get("message", {}).get("content", "")

            # Usage (may not be present for all models/providers)
            usage = resp.get("usage", {}) or {}
            total_tokens = usage.get("total_tokens", None)
            prompt_tokens = usage.get("prompt_tokens", None)
            completion_tokens = usage.get("completion_tokens", None)

            return {
                "model": model_name,
                "text": text.strip(),
                "latency_s": latency,
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "error": None,
            }
        except Exception as e:
            latency = time.perf_counter() - start
            return {
                "model": model_name,
                "text": "",
                "latency_s": latency,
                "total_tokens": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "error": str(e),
            }

    # Run calls (parallel)
    st.info(f"Calling {len(models)} model(s)...")
    results = []
    with st.spinner("Calling models..."):
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            future_to_model = {ex.submit(call_model, m): m for m in models}
            for future in as_completed(future_to_model):
                res = future.result()
                results.append(res)

    # Sort results by latency ascending
    results = sorted(results, key=lambda r: r["latency_s"])

    # Display per-model outputs
    cols = st.columns(len(results) if len(results) <= 3 else 3)
    st.subheader("Model responses")
    for i, r in enumerate(results):
        col = cols[i % len(cols)]
        with col:
            st.markdown(f"### {r['model']}")
            if r["error"]:
                st.error(f"Error: {r['error']}")
            else:
                st.write(r["text"])
                st.caption(f"Latency: {r['latency_s']:.2f} s • Tokens (total/prompt/comp): "
                           f"{r['total_tokens']}/{r['prompt_tokens']}/{r['completion_tokens']}")
    st.markdown("---")

    # Summary table
    df = pd.DataFrame([
        {
            "model": r["model"],
            "latency_s": r["latency_s"],
            "total_tokens": (r["total_tokens"] if r["total_tokens"] is not None else np.nan),
            "error": r["error"],
            "text_preview": (r["text"][:200] + "..." if r["text"] and len(r["text"]) > 200 else r["text"])
        } for r in results
    ])
    st.subheader("Summary metrics")
    st.dataframe(df[["model", "latency_s", "total_tokens", "error", "text_preview"]].sort_values("latency_s"))

    # Bar charts for latency and tokens
    st.subheader("Visual metrics")
    chart_df = df.drop(columns=["error", "text_preview"]).copy()
    chart_df["latency_s"] = chart_df["latency_s"].astype(float)
    chart_df["total_tokens"] = chart_df["total_tokens"].astype(float)

    lat_chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("model:N", sort="-y"),
        y=alt.Y("latency_s:Q", title="Latency (s)"),
        color=alt.Color("model:N")
    ).properties(width=400, height=250, title="Latency by model")
    st.altair_chart(lat_chart, use_container_width=True)

    token_chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("model:N", sort="-y"),
        y=alt.Y("total_tokens:Q", title="Total tokens"),
        color=alt.Color("model:N")
    ).properties(width=400, height=250, title="Token usage by model")
    st.altair_chart(token_chart, use_container_width=True)

    # Compute embeddings for each response and pairwise similarities
    st.subheader("Pairwise semantic similarity between model answers")
    texts = [r["text"] if r["text"] else "" for r in results]
    model_names = [r["model"] for r in results]

    # If all responses empty -> skip
    if all(t.strip() == "" for t in texts):
        st.warning("No textual responses to compute similarity.")
    else:
        # Create embeddings (use small embedding model)
        try:
            emb_model = "text-embedding-3-small"
            embs = []
            for t in texts:
                if t.strip() == "":
                    embs.append(np.zeros(1536))  # placeholder of right size may vary; we'll compute gracefully
                else:
                    e = openai.Embedding.create(model=emb_model, input=t)
                    emb = np.array(e["data"][0]["embedding"], dtype=float)
                    embs.append(emb)

            # Normalize and compute cosine similarity matrix
            def cosine_sim(a, b):
                if np.all(a == 0) or np.all(b == 0):
                    return float("nan")
                return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

            sim_mat = np.zeros((len(embs), len(embs)), dtype=float)
            for i in range(len(embs)):
                for j in range(len(embs)):
                    sim_mat[i, j] = cosine_sim(embs[i], embs[j])

            sim_df = pd.DataFrame(sim_mat, index=model_names, columns=model_names)
            st.dataframe(sim_df.style.format("{:.3f}"))
        except Exception as e:
            st.error(f"Embedding / similarity step failed: {e}")

    st.success("Done ✅")
