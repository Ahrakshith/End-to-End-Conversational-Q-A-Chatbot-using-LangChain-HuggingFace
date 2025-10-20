# app_new_openai.py
import os, time
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI

load_dotenv()
st.set_page_config(page_title="Q&A (OpenAI v1)")
st.title("Simple Q&A — OpenAI v1 client")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing (set in env or .env)")
    st.stop()

# create client (it reads api_key param or env var)
client = OpenAI(api_key=OPENAI_API_KEY)

with st.sidebar:
    model = st.selectbox("Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max tokens", 50, 2000, 300)

prompt = st.text_area("Ask a question", height=150)
if st.button("Send"):
    if not prompt.strip():
        st.warning("Type a question")
    else:
        st.info("Calling model...")
        try:
            start = time.perf_counter()
            # Chat Completions API (new client)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            elapsed = time.perf_counter() - start

            # old-style message text extraction:
            text = completion.choices[0].message.content
            st.markdown("### Response")
            st.write(text)
            st.caption(f"Model: {model} • Time: {elapsed:.2f}s")
        except Exception as e:
            st.error("OpenAI API error")
            st.exception(e)
