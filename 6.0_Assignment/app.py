
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/malcolmosh/goodbooks-10k-extended/master/books_enriched.csv"
    df = pd.read_csv(url)
    df = df[['title', 'authors', 'description']].dropna()
    return df

# Embed descriptions
@st.cache_resource
def embed_descriptions(descriptions):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(descriptions, show_progress_bar=True)
    return embeddings

# Search function
def search_books(query, embeddings, df, k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vec = model.encode([query])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    D, I = index.search(query_vec, k)
    return df.iloc[I[0]]

# Generate GPT response
def generate_gpt_response(prompt):
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

# Streamlit UI
def main():
    st.title("📚 Book Recommendation Chatbot")
    st.write("Ask for book recommendations based on your interests!")

    df = load_data()
    embeddings = embed_descriptions(df['description'].tolist())

    query = st.text_input("What kind of book are you looking for?")
    if query:
        results = search_books(query, embeddings, df)
        st.subheader("Top Recommendations:")
        prompt = "Here are some book recommendations based on your query:\n"
        for _, row in results.iterrows():
            prompt += f"**{row['title']}** by *{row['authors']}*\n{row['description']}\n\n"
        gpt_response = generate_gpt_response(prompt)
        st.write(gpt_response)

if __name__ == "__main__":
    main()
