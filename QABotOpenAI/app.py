# app.py (robust, avoids TypeError when env var is missing)
import os
import time
from dotenv import load_dotenv
import streamlit as st

load_dotenv()  # optional: load values from .env

st.set_page_config(page_title="Simple Q&A (OpenAI)", layout="wide")
st.title("Simple Q&A Chatbot (OpenAI) — robust")

# Safely copy LANGCHAIN env only if present (avoid assigning None)
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key is not None:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# Optionally set or copy other LangChain envs only if present
langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2")
if langchain_tracing is not None:
    os.environ["LANGCHAIN_TRACING_V2"] = langchain_tracing

# Set project name only if provided
langchain_project = os.getenv("LANGCHAIN_PROJECT")
if langchain_project is not None:
    os.environ["LANGCHAIN_PROJECT"] = langchain_project

# Load OpenAI API key from environment or .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Set it in your environment or in a .env file.")
    st.stop()

# Import openai and set key, with UI-visible error handling
try:
    import openai
    openai.api_key = OPENAI_API_KEY
    st.write(f"openai package OK")
except Exception as e:
    st.error("Failed to import or configure openai.")
    st.exception(e)
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    engine = st.selectbox("Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    max_tokens = st.slider("Max tokens", 50, 2000, 300, 50)
    st.markdown("---")
    st.markdown("Note: OPENAI_API_KEY is read from your environment; not shown here.")

st.write("Ask a question:")
question = st.text_area("", height=140, placeholder="Type something like: Explain the difference between X and Y")

if st.button("Send"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        st.info("Calling model...")
        try:
            start = time.perf_counter()
            resp = openai.ChatCompletion.create(
                model=engine,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            elapsed = time.perf_counter() - start

            # Extract answer text safely
            text = ""
            choices = resp.get("choices")
            if choices and len(choices) > 0:
                # join multiple choices' content if present
                text = "".join([c.get("message", {}).get("content", "") for c in choices])
            else:
                text = resp.get("message", {}).get("content", "")

            st.markdown("### Response")
            st.write(text)
            st.caption(f"Model: {engine} • Time: {elapsed:.2f}s • Usage: {resp.get('usage', {})}")

        except Exception as e:
            st.error("Error calling OpenAI API")
            st.exception(e)
else:
    st.info("Type a question and press Send.")
