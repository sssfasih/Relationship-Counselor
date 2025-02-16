import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Initialize OpenAI API
api_key = ""  # Replace with your API key
base_url = "https://api.aimlapi.com/v1"
api = OpenAI(api_key=api_key, base_url=base_url)

# Initialize Sentence Transformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS Index
dimension = 384  # Embedding dimension of the model
index = faiss.IndexFlatL2(dimension)

# Function to chunk text
def chunk_text(text, max_length=500):
    words = text.split()
    chunks = []
    chunk = []
    for word in words:
        if len(" ".join(chunk)) + len(word) <= max_length:
            chunk.append(word)
        else:
            chunks.append(" ".join(chunk))
            chunk = [word]
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

# Function to embed text and add to FAISS index
def embed_and_store(chunks):
    embeddings = embedding_model.encode(chunks)
    index.add(embeddings)

# Query handling
def query_llm(prompt):
    completion = api.chat.completions.create(
        model="deepseek-ai/deepseek-llm-67b-chat",
        messages=[
            {"role": "system", "content": "You are a relationship counselor. Analyze the given WhatsApp conversation and provide insights on potential red flags, toxicity, and room for improvement in behavior. Every response must start by rating the overall chat toxicity out of 10."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=350,
    )
    return completion.choices[0].message.content

# Streamlit App
st.title("AI Relationship Counsellor")

uploaded_file = st.file_uploader("Upload a text file of your WhatsApp chat", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    st.write("Chat Extracted Successfully!")

    # Chunk and embed text
    chunks = chunk_text(text)
    embed_and_store(chunks)

    # Query Interface
    user_query = st.text_input("Ask a question about your relationship:")
    if user_query:
        # Embed query and search FAISS
        query_embedding = embedding_model.encode([user_query])
        distances, indices = index.search(query_embedding, k=5)  # Top 5 results
        relevant_chunks = [chunks[i] for i in indices[0]]

        # Combine chunks for context
        context = " ".join(relevant_chunks)
        final_prompt = f"Context: {context}\n\nQuestion: {user_query}"

        # Get response from AI model
        response = query_llm(final_prompt)
        st.write("### AI Analysis")
        st.write(response) 
