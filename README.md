🤖 OwnChatbot — PDF RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot that allows users to upload a PDF document and ask questions about its content.

The application extracts text from the PDF, splits it into smaller chunks, converts the chunks into vector embeddings, stores them in FAISS, retrieves the most relevant content, and uses Google Gemini to generate the final answer.

🔄 RAG Workflow
1. PDF Upload

The user uploads a PDF through the Streamlit interface.

2. Text Extraction

pdfplumber extracts readable text from each page.

3. Text Chunking

The extracted text is divided into smaller chunks using RecursiveCharacterTextSplitter.

RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200
)
4. Embeddings

Each text chunk is converted into a numerical vector using:

gemini-embedding-001
5. Vector Storage

The generated embeddings are stored in a FAISS vector database.

6. Retrieval

When the user asks a question, the question is compared with the stored vectors and the most relevant document chunks are retrieved.

7. Generation

The retrieved context is passed to Gemini, which generates the final response.

📦 Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/rag-chatbot.git

Move into the project directory:

cd rag-chatbot

Create a virtual environment:

python -m venv venv

Activate the virtual environment on Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

🔑 API Key Configuration

Create the following file:

.streamlit/secrets.toml

Add your Gemini API key:

GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"

The API key should never be hardcoded in the Python source code or committed to GitHub.

Make sure .streamlit/secrets.toml is included in .gitignore.

▶️ Run the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser.
