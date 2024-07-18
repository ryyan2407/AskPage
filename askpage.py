import streamlit as st
import os
from groq import Groq
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_website_content(url, chunk_size=3):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()

        paragraphs = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 50]

        chunks = []
        for i in range(0, len(paragraphs), chunk_size):
            chunk = " ".join(paragraphs[i:i+chunk_size])
            chunks.append(chunk)
        
        return chunks
    except Exception as e:
        st.error(f"Error fetching website content: {str(e)}")
        return None

def create_embeddings(text_chunks):
    return model.encode(text_chunks)

def get_relevant_content(query, text_chunks, embeddings, top_k=2):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [text_chunks[i] for i in top_indices]

def query_groq(prompt, context, url, chat_history):
    system_message = f"""You are an AI assistant specialized in analyzing and answering questions about web content. 
    Your task is to provide accurate, relevant, and insightful responses based on the following context from the website {url}:

    Guidelines:
    1. Analyze the context thoroughly before answering.
    2. If the answer is explicitly stated in the context, provide it directly.
    3. If the answer requires inference, clearly state that you're making an inference based on the available information.
    4. If the question cannot be answered based solely on the given context, say so and explain why.
    5. When referencing information from the context, use direct quotes and cite them at the end of your response like this: "Quote" (Source: relevant part of the context)
    6. If asked about topics not covered in the context, politely explain that you can only answer based on the provided website content.
    7. Maintain a professional and helpful tone throughout the interaction.
    8. If appropriate, suggest related questions that the user might find interesting based on the context.
    9. Consider the chat history when formulating your response.

    {context}

    Remember, your knowledge is limited to the provided context. Do not invent or assume information not present in the given text."""

    messages = [
        {"role": "system", "content": system_message},
    ]
    
  
    for message in chat_history:
        messages.append({"role": "user", "content": message["user"]})
        messages.append({"role": "assistant", "content": message["assistant"]})
    
   
    messages.append({"role": "user", "content": prompt})

    chat_completion = groq_client.chat.completions.create(
        messages=messages,
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content

def main():
    st.title("AskPage")

    if "url" not in st.session_state:
        st.session_state.url = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = []
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "question_asked" not in st.session_state:
        st.session_state.question_asked = False

    url = st.text_input("Enter a website URL:", value=st.session_state.url)

    if url != st.session_state.url:
        st.session_state.url = url
        st.session_state.chat_history = []
        if url:
            with st.spinner("Fetching and processing website content..."):
                content = get_website_content(url, chunk_size=3)
            
            if content:
                st.success("Website content fetched successfully!")
                st.session_state.text_chunks = content
                st.session_state.embeddings = create_embeddings(content)
            else:
                st.error("Failed to fetch website content. Please check the URL and try again.")
                st.session_state.text_chunks = []
                st.session_state.embeddings = None

    if st.session_state.text_chunks:
        for i, message in enumerate(st.session_state.chat_history):
            st.text_area(f"Conversation {i+1}:", 
                         value=f"You: {message['user']}\n\nAssistant: {message['assistant']}", 
                         height=200, 
                         key=f"conversation_{i}",
                         disabled=True)
            st.markdown("---")

        user_question = st.text_input("Ask a question about the website:", key="user_input")
        
        if st.button("Send") or (user_question and not st.session_state.question_asked):
            if user_question:
                st.session_state.question_asked = True
                with st.spinner("Generating response..."):
                    relevant_chunks = get_relevant_content(user_question, st.session_state.text_chunks, st.session_state.embeddings)
                    context = "\n".join(relevant_chunks)
                    response = query_groq(user_question, context, url, st.session_state.chat_history)
                
                st.session_state.chat_history.append({"user": user_question, "assistant": response})
                
                st.text_area("Latest Conversation:", 
                             value=f"You: {user_question}\n\nAssistant: {response}", 
                             height=200, 
                             key="latest_conversation",
                             disabled=True)

                st.experimental_rerun()

        if not user_question:
            st.session_state.question_asked = False

if __name__ == "__main__":
    main()