# 🤖 End-to-End Q&A Chatbot using OpenAI and LangChain

An interactive **Generative AI Q&A Chatbot** built with **Streamlit**, **OpenAI GPT models**, and **LangChain**, designed to demonstrate a full end-to-end large-language-model (LLM) pipeline — from prompt engineering and response generation to analytics and deployment.

> 🧠 *Built as part of GNI Project (Feb 2025 – May 2025)*  
> Includes both a **Simple Q&A Mode** and an **Advanced Multi-Model Comparison Mode** with latency and semantic-similarity metrics.

---

## 🚀 Features

| Category | Description |
|-----------|-------------|
| 💬 **Simple Q&A Mode** | Ask any question and get a concise answer from the selected OpenAI GPT model. |
| ⚙️ **Model Comparison Mode** | Run multiple models in parallel, view each model’s output, latency, and token usage. |
| 📊 **Analytics Dashboard** | Displays bar charts for latency and token counts, plus pairwise semantic-similarity heatmaps. |
| 🔐 **Secure Key Handling** | Uses environment variables or Streamlit Secrets for API key management — no key exposed in UI. |
| ☁️ **Deployable Anywhere** | Easily deployable to **Streamlit Cloud** or any Python environment. |
| 🧩 **Extensible Design** | Ready to integrate with retrieval (RAG) modules, databases, or conversation memory. |

---

## 🧱 Architecture Overview

User (Streamlit UI)
        │
        ▼
LangChain / OpenAI Client (LLM)
        │
        ▼
Response Parser  →  Metrics Collector  →  Streamlit Visualizer


Frontend: Streamlit (interactive web UI)

Backend: OpenAI GPT models (gpt-4o, gpt-4-turbo, gpt-3.5-turbo)

Framework: LangChain for prompt templates & parsing (optional)

Environment Management: .env + Streamlit Secrets

 ##Tech Stack

Frontend: Streamlit

Backend / LLM: OpenAI GPT models (via openai Python client v1+)

Frameworks: LangChain, Python

Utilities: NumPy, Pandas, Altair (for analytics & visualization)

Deployment: Streamlit Cloud

##Environment: .env or Streamlit Secrets for API keys

🛠️ Setup Instructions
1. Clone the repository
git clone https://github.com/<your-username>/QABotOpenAI.git
cd QABotOpenAI

2. Create a virtual environment
python -m venv myenv
source myenv/bin/activate    # macOS/Linux
myenv\Scripts\activate       # Windows

3. Install dependencies
pip install -r requirements.txt

4. Create a .env file
OPENAI_API_KEY=sk-your-openai-key

5. Run the app
streamlit run app.py


Then open http://localhost:8501


##Deploy on Streamlit Cloud

Push to GitHub

On Streamlit Cloud
 → New App

In Secrets, add:

OPENAI_API_KEY = "sk-your-openai-key"


Deploy 🎉

##Project Structure
QABotOpenAI/
│
├── app.py                # Main Streamlit app
├── requirements.txt      # Python dependencies
├── .env                  # (optional) Local environment variables
└── README.md             # Documentation

## Future Enhancements

🗃️ Conversation history (SQLite)

📚 RAG (document-based Q&A)

💵 Token cost tracking

☁️ Support for local LLMs (Ollama / Hugging Face)

##Author

Rakshith
GNI Project — 2025

End-to-End Q&A Chatbot using OpenAI and LangChain
Built to demonstrate Generative AI engineering — from model orchestration to deployment.

