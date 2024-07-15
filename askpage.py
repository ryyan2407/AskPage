import streamlit as st
import os
from exa_py import Exa
from groq import Groq
from bs4 import BeautifulSoup
import requests


exa = Exa(os.environ.get("EXA_API_KEY"))
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_website_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except:
        return None

def get_relevant_snippets(url, content):
    try:
        response = exa.search_and_contents(
            f"content from {url}",
            use_autoprompt=True,
            num_results=5,
            text=True,
            highlights=True
        )
        
        relevant_snippets = []
        for result in response.results:
            if result.highlights:
                relevant_snippets.extend(result.highlights)
        
        if relevant_snippets:
            return " ".join(relevant_snippets)
        else:
            return content[:1000]
    except Exception as e:
        st.warning(f"Error fetching snippets from Exa: {str(e)}. Using default content.")
        return content[:1000]


def query_groq(prompt, context, url, chat_history):
    system_message = f"""You are an AI assistant specialized in analyzing and answering questions about web content. 
    Your task is to provide accurate, relevant, and insightful responses based on the following context from the website {url}:

    Guidelines:
    1. Analyze the context thoroughly before answering.
    2. If the answer is explicitly stated in the context, provide it directly.
    3. If the answer requires inference, clearly state that you're making an inference based on the available information.
    4. If the question cannot be answered based solely on the given context, say so and explain why.
    5. Provide specific references or quotes from the context to support your answers when applicable.
    6. If asked about topics not covered in the context, politely explain that you can only answer based on the provided website content.
    7. Maintain a professional and helpful tone throughout the interaction.
    8. If appropriate, suggest related questions that the user might find interesting based on the context.
    9. Consider the chat history when formulating your response.

    {context}

    Remember, your knowledge is limited to the provided context. Do not invent or assume information not present in the given text."""

    messages = [
        {"role": "system", "content": system_message},
    ]
    
    # Add chat history
    for message in chat_history:
        messages.append({"role": "user", "content": message["user"]})
        messages.append({"role": "assistant", "content": message["assistant"]})
    
    # Add current prompt
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
    if "context" not in st.session_state:
        st.session_state.context = ""

    url = st.text_input("Enter a website URL:", value=st.session_state.url)

    if url != st.session_state.url:
        st.session_state.url = url
        st.session_state.chat_history = []
        if url:
            with st.spinner("Fetching website content..."):
                content = get_website_content(url)
            
            if content:
                st.success("Website content fetched successfully!")
                st.session_state.context = get_relevant_snippets(url, content)
            else:
                st.error("Failed to fetch website content. Please check the URL and try again.")
                st.session_state.context = ""

    if st.session_state.context:
        user_question = st.text_input("Ask a question about the website:")
        
        if user_question:
            with st.spinner("Generating response..."):
                response = query_groq(user_question, st.session_state.context, url, st.session_state.chat_history)
            
            st.session_state.chat_history.append({"user": user_question, "assistant": response})

        # Display chat history
        for message in st.session_state.chat_history:
            st.write("You:", message["user"])
            st.write("Assistant:", message["assistant"])
            st.write("---")

if __name__ == "__main__":
    main()