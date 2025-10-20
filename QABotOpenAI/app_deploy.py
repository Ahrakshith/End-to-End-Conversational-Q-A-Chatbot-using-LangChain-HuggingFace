# app.py
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from dotenv import load_dotenv

# --- Use the new OpenAI client ---
# pip install openai>=1.0.0
from openai import OpenAI

# Load local .env if present (optional for local dev)
load_dotenv()

# --- App config ---
st.set_page_config(page_title="Multi-Model Q&A + Metrics", layout="wide")
st.title("Multi-Model Q&A Chatbot — Responses + Metrics")

st.markdown(
    """
This app calls multiple OpenAI models in parallel for the same question and displays:
- each model's answer  
- latency (time taken)  
- token usage (when available)  
- pairwise semantic similarity between answers using embeddings  
\n**Note:** `OPENAI_API_KEY` must be set in your environment or in a `.env` file (not shown in the UI).
"""
)

# --- Read API key from env safely ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY not found. Set OPENAI_API_KEY in your environment (or create a `.env` file)."
    )
    st.stop()

# Create OpenAI client
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error("Failed to create OpenAI client.")
    st.exception(e)
    st.stop()

# --- Sidebar settings (no API key visible) ---
with st.sidebar:
    st.header("Settings")
    model_choices = [
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        # add / remove models you have access to
    ]
    selected_models = st.multiselect(
        "Choose models to compare",
        options=model_choices,
        default=["gpt-4o", "gpt-3.5-turbo"],
        help="Pick 1..N models to call for the same question."
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    max_tokens = st.slider("Max tokens (response)", 50, 2000, 300, 50)
    concurrency = st.slider(
        "Parallel requests (threads)", 1, min(8, max(1, len(selected_models) or 1)), value=min(4, max(1, len(selected_models) or 1))
    )
    st.markdown("---")
    st.markdown("Tip: Set OPENAI_API_KEY as an environment variable (or in `.env` locally).")

# --- Main input ---
st.write("Ask a question to compare model outputs")
question = st.text_area("Question", height=160, placeholder="e.g. Explain the difference between X and Y in simple terms")

if st.button("Run comparison"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()
    if not selected_models:
        st.warning("Please select at least one model in the sidebar.")
        st.stop()

    system_prompt = "You are a helpful assistant. Answer concisely and clearly."

    # Helper to call one model and return metrics
    def call_model(model_name: str) -> Dict[str, Any]:
        start_time = time.perf_counter()
        try:
            # New OpenAI v1 client: chat completions via client.chat.completions.create
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
            latency = time.perf_counter() - start_time

            # Extract text safely
            text = ""
            try:
                # The response object is often a dataclass-like object; handle multiple possible shapes
                choices = getattr(resp, "choices", None) or resp.get("choices", None)
                if choices and len(choices) > 0:
                    first = choices[0]
                    # Access .message.content or dict path
                    msg = getattr(first, "message", None) or first.get("message", {})
                    content = getattr(msg, "content", None) or msg.get("content", "")
                    text = content or ""
                else:
                    # fallback
                    text = getattr(resp, "output_text", None) or resp.get("output_text", "") or ""
            except Exception:
                text = str(resp)

            # Usage (may be missing for some models / responses)
            usage = {}
            try:
                usage_attr = getattr(resp, "usage", None) or resp.get("usage", None)
                if usage_attr:
                    usage = dict(usage_attr)
            except Exception:
                usage = {}

            return {
                "model": model_name,
                "text": text.strip(),
                "latency_s": latency,
                "usage": usage,
                "error": None,
            }
        except Exception as e:
            latency = time.perf_counter() - start_time
            return {
                "model": model_name,
                "text": "",
                "latency_s": latency,
                "usage": {},
                "error": str(e),
            }

    # Run model calls in parallel
    st.info(f"Calling {len(selected_models)} model(s) ...")
    results: List[Dict[str, Any]] = []
    with st.spinner("Calling models..."):
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(call_model, m): m for m in selected_models}
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)

    # sort by latency for display
    results = sorted(results, key=lambda r: r["latency_s"])

    # Display results
    st.subheader("Model responses")
    cols = st.columns(len(results) if len(results) <= 3 else 3)
    for i, r in enumerate(results):
        col = cols[i % len(cols)]
        with col:
            st.markdown(f"### {r['model']}")
            if r["error"]:
                st.error(f"Error: {r['error']}")
            else:
                st.write(r["text"] or "_(empty response)_")
                usage = r.get("usage", {}) or {}
                total = usage.get("total_tokens") or usage.get("total") or usage.get("total_tokens", None)
                prompt_t = usage.get("prompt_tokens", None)
                comp_t = usage.get("completion_tokens", None)
                # Some model responses may not have token fields; show usage dict if none of the expected keys exist
                if total or prompt_t or comp_t:
                    st.caption(f"Latency: {r['latency_s']:.2f}s • Tokens: total={total} prompt={prompt_t} completion={comp_t}")
                else:
                    st.caption(f"Latency: {r['latency_s']:.2f}s • Usage: {usage or 'n/a'}")

    st.markdown("---")

    # Summary table
    df = pd.DataFrame([
        {
            "model": r["model"],
            "latency_s": r["latency_s"],
            "total_tokens": (r.get("usage", {}) or {}).get("total_tokens", np.nan),
            "error": r["error"],
            "text_preview": (r["text"][:200] + "...") if r["text"] and len(r["text"]) > 200 else r["text"]
        }
        for r in results
    ])
    st.subheader("Summary metrics")
    st.dataframe(df[["model", "latency_s", "total_tokens", "error", "text_preview"]].sort_values("latency_s"))

    # Charting
    st.subheader("Visual metrics")
    chart_df = df.copy()
    chart_df["latency_s"] = chart_df["latency_s"].astype(float)
    # Replace NaN with 0 for charting
    chart_df["total_tokens"] = pd.to_numeric(chart_df["total_tokens"], errors="coerce").fillna(0)

    lat_chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("model:N", sort="-y"),
        y=alt.Y("latency_s:Q", title="Latency (s)"),
        color=alt.Color("model:N")
    ).properties(width=600, height=250, title="Latency by model")
    st.altair_chart(lat_chart, use_container_width=True)

    token_chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("model:N", sort="-y"),
        y=alt.Y("total_tokens:Q", title="Total tokens"),
        color=alt.Color("model:N")
    ).properties(width=600, height=250, title="Token usage by model")
    st.altair_chart(token_chart, use_container_width=True)

    # Compute embeddings for semantic similarity
    st.subheader("Pairwise semantic similarity (answers)")
    texts = [r["text"] or "" for r in results]
    model_names = [r["model"] for r in results]

    if all(not t.strip() for t in texts):
        st.warning("No textual responses to compute similarity.")
    else:
        try:
            emb_model = "text-embedding-3-small"
            embeddings = []
            for t in texts:
                if not t.strip():
                    embeddings.append(None)
                    continue
                emb_resp = client.embeddings.create(model=emb_model, input=t)
                # response shape: emb_resp.data[0].embedding
                emb_vec = None
                try:
                    emb_vec = np.array(emb_resp.data[0].embedding, dtype=float)
                except Exception:
                    # fallback if dict-like
                    d0 = emb_resp.get("data", [])[0]
                    emb_vec = np.array(d0.get("embedding", []), dtype=float)
                embeddings.append(emb_vec)

            # compute cosine similarities
            n = len(embeddings)
            sim_mat = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(n):
                    a = embeddings[i]
                    b = embeddings[j]
                    if a is None or b is None or len(a) == 0 or len(b) == 0:
                        sim = np.nan
                    else:
                        denom = (np.linalg.norm(a) * np.linalg.norm(b))
                        sim = float(np.dot(a, b) / denom) if denom != 0 else np.nan
                    sim_mat[i, j] = sim

            sim_df = pd.DataFrame(sim_mat, index=model_names, columns=model_names)
            st.dataframe(sim_df.style.format("{:.3f}"))
        except Exception as e:
            st.error("Embedding/similarity step failed")
            st.exception(e)

    st.success("Done ✅")
else:
    st.info("Enter a question and press 'Run comparison' to call selected models.")
