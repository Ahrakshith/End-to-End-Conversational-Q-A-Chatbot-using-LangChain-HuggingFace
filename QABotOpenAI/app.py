import os
import time
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI   # new v1 client

# --- Load environment variables (.env for local dev) ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Simple Q&A Chatbot", layout="centered")
st.title("Simple Q&A Chatbot (OpenAI v1)")

# --- Validate API key ---
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found. Set it in your environment or .env file.")
    st.stop()

# --- Create client ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Sidebar settings ---
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max tokens", 50, 2000, 300, 50)

# --- Main UI ---
st.write("Ask a question:")
question = st.text_area("", height=150, placeholder="Type your question here...")

if st.button("Send"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    st.info("Calling model...")
    try:
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = time.perf_counter() - start

        answer = response.choices[0].message.content
        st.markdown("### Response")
        st.write(answer)
        st.caption(f"Model: {model} • Time: {elapsed:.2f}s")

    except Exception as e:
        st.error("Error calling OpenAI API:")
        st.exception(e)
else:
    st.info("Type a question and click **Send** to get an answer.")
