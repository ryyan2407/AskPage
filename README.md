# AskPage: Your AI-Powered Web Explorer

## Introduction

AskPage is a cutting-edge web application that revolutionizes how you interact with online content. Powered by advanced AI and RAG (Retrieval-Augmented Generation) technology, AskPage allows you to ask questions about any website and receive intelligent, context-aware responses.

## Features

- **Web Content Analysis**: Simply input a URL, and AskPage will scrape and analyze the content.
- **Intelligent Q&A**: Ask any question about the website, and get accurate, contextual answers.
- **RAG Technology**: Utilizes state-of-the-art Retrieval-Augmented Generation for precise information retrieval.
- **Dynamic Context Understanding**: The AI adapts its responses based on the specific content of each website.
- **User-Friendly Interface**: Clean, intuitive Streamlit-based UI for seamless interaction.
- **Conversation History**: Keep track of your Q&A session with a chat-like interface.

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Web Scraping**: BeautifulSoup4
- **Embeddings**: Sentence Transformers
- **LLM Integration**: Groq API
- **Text Processing**: NumPy, scikit-learn

## Getting Started

1. Clone the repository:
   git clone https://github.com/yourusername/askpage.git
2. Install dependencies:
   pip install -r requirements.txt
3. Set up your environment variables:
- `GROQ_API_KEY`: Your Groq API key
4. Run the application:
   streamlit run askpage.py

## How to Use

1. Enter a website URL in the provided input field.
2. Wait for AskPage to process the content (this usually takes a few seconds).
3. Once loaded, type your question about the website in the chat input.
4. Click 'Send' or press Enter to get your AI-generated response.
5. Continue the conversation or load a new website to explore!


## Acknowledgements

- [Streamlit](https://streamlit.io/) for the web application framework
- [Groq](https://groq.com/) for AI language processing
