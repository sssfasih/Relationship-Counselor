# 💬 AI Relationship Advisor

<img src="digital-illustration.png" alt="AI Relationship Counsellor" width="500"/>

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


## 🧪 Installation & Local Run

Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
```

Set your Groq API key (via terminal or `.env`):

```bash
export GROQ_API="your_api_key_here"
```

Then run the app:

```bash
streamlit run app.py
```

---

## 🌐 Deployment

This app is deployed on Hugging Face Spaces:  
👉 [RelationshipCounsellor on Hugging Face](https://huggingface.co/spaces/sssfasihieee/RelationshipCounsellor)

To deploy your own:

1. Push to a Hugging Face Space with `Streamlit` runtime.
2. Set your `GROQ_API` secret via **Settings > Secrets**.

Reference:  
📖 https://huggingface.co/docs/hub/spaces-config-reference

---

## 🔐 Environment Variables

| Variable     | Description            |
|--------------|------------------------|
| `GROQ_API`   | Your Groq API key      |

---

## 📌 License

MIT License. Feel free to fork and improve!

---

> Built with ❤️ by Syed Fasih Uddin  
