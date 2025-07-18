# 💬 AI Relationship Advisor

![AI Relationship Counsellor](digital_illustration.png)

This is an AI-powered relationship counselling app built using **Streamlit** and deployed on **Hugging Face Spaces**. It leverages **Retrieval-Augmented Generation (RAG)** to provide contextual, personalized, and insightful advice based on WhatsApp chat uploads and user questions.


---

## 🚀 Features

- **📄 Upload WhatsApp Chats:** Upload `.txt` chat files for analysis.
- **🧠 RAG-Powered Contextual Answers:** Uses FAISS and Sentence Transformers to ground AI responses in real chat context.
- **🤖 LLM-Based Insights:** AI analyzes red flags, toxic behavior, and relationship health using **Groq’s LLaMA 3.3-70B Versatile** model.
- **🖼️ Clean UI:** Built with Streamlit for an interactive and smooth user experience.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Streamlit** | UI framework |
| **FAISS (CPU)** | Vector search for semantic retrieval |
| **Sentence Transformers** | Embedding generation (`all-MiniLM-L6-v2`) |
| **Groq API** | Fast, low-latency inference with `llama-3.3-70b-versatile` |
| **Hugging Face Spaces** | Deployment platform |

---

## Installation & Running Locally

To run this app locally, install the dependencies and start the Streamlit server:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

This app is deployed on Hugging Face Spaces and can be accessed [here](https://huggingface.co/spaces/sssfasihieee/RelationshipCounsellor).

---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
