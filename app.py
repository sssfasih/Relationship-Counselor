import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Initialize OpenAI API
api_key = "f614cad0fabe42bd8f287a921066b771"  # Replace with your API key
base_url = "https://api.aimlapi.com/v1"
api = OpenAI(api_key=api_key, base_url=base_url)

# Initialize Sentence Transformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS Index (using L2 distance)
dimension = 384  # Embedding dimension for 'all-MiniLM-L6-v2'
index = faiss.IndexFlatL2(dimension)

# Function to chunk text into manageable pieces
def chunk_text(text, max_length=500):
    words = text.split()
    chunks = []
    chunk = []
    for word in words:
        # Check if adding the next word would exceed the max length
        if len(" ".join(chunk)) + len(word) <= max_length:
            chunk.append(word)
        else:
            chunks.append(" ".join(chunk))
            chunk = [word]
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

# Function to embed text chunks and add them to the FAISS index
def embed_and_store(chunks):
    embeddings = embedding_model.encode(chunks)
    index.add(embeddings)

# Function to query the LLM using the retrieved context
def query_llm(prompt):
    completion = api.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=256,
    )
    return completion.choices[0].message.content

# Streamlit App Interface
st.title("RAG-based Document Query App")
st.write("Upload a plain text file and ask questions about its content.")

# Change file uploader to accept only .txt files
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    # Read and decode the uploaded text file
    text = uploaded_file.read().decode("utf-8")
    st.success("Text file loaded successfully!")

    # Chunk the text and build the FAISS index
    chunks = chunk_text(text)
    embed_and_store(chunks)
    st.write(f"{len(chunks)} chunks added to the FAISS index.")

    # Query Interface
    user_query = st.text_input("Ask a question about the document:")
    if user_query:
        # Embed the query and search the FAISS index for top 5 similar chunks
        query_embedding = embedding_model.encode([user_query])
        distances, indices = index.search(query_embedding, k=5)
        relevant_chunks = [chunks[i] for i in indices[0]]

        # Combine retrieved chunks to create context for the LLM
        context = " ".join(relevant_chunks)
        final_prompt = f"Context: {context}\n\nQuestion: {user_query}"

        # Get and display the answer from the LLM
        response = query_llm(final_prompt)
        st.subheader("Answer")
        st.write(response)
