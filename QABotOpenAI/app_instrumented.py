# app_instrumented_fixed.py
import os
import sys
import traceback
import streamlit as st

st.set_page_config(page_title="Instrumented Debug App (fixed)", layout="wide")
st.title("Instrumented Debug App — fixed")

st.markdown("**Stage 0:** basic env info")
try:
    st.write(f"Python: {sys.version.split()[0]}")
    st.write(f"Streamlit version: {st.__version__}")
except Exception as e:
    st.error("Stage 0 failed")
    st.exception(e)
    raise

st.markdown("**Stage 1:** Attempting light imports (openai/langchain) — errors shown below")
import_errors = []

# openai
try:
    import openai
    st.write("openai imported OK")
except Exception:
    import_errors.append(("openai", traceback.format_exc()))
    st.text_area("openai import error", traceback.format_exc(), height=200)

# try langchain variants
try:
    try:
        from langchain.prompts import ChatPromptTemplate
        st.write("imported: langchain.prompts.ChatPromptTemplate")
    except Exception:
        from langchain_core.prompts import ChatPromptTemplate
        st.write("imported: langchain_core.prompts.ChatPromptTemplate")
except Exception:
    import_errors.append(("langchain.prompts / langchain_core.prompts", traceback.format_exc()))
    st.text_area("langchain import error", traceback.format_exc(), height=300)

st.markdown("**Stage 2:** Environment variables (masked)")
keys = ["OPENAI_API_KEY", "LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_PROJECT"]
for k in keys:
    val = os.getenv(k)
    st.write(f"{k}: {'SET' if val else 'NOT SET'}")

st.markdown("**Stage 3:** Minimal interactive test (no API calls)")
question = st.text_input("Type a question (no LLM calls in this test):")
if st.button("Show echo"):
    st.write("You typed:", question)

st.markdown("---")
st.info("If you still get a blank page after this loads, open DevTools -> Console and paste any console errors.")
