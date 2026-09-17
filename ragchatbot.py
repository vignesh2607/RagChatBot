import os
import pdfplumber as pdf
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Page configuration
st.set_page_config(
    page_title="ownchatbot",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("📁 Documents")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")


# -----------------------------
# Main Page
# -----------------------------
st.title("🤖 ownchatbot")

st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Ask anything about your documents"
    "</p>",
    unsafe_allow_html=True
)

# Add some space
st.write("")
st.write("")


# -----------------------------
# Center Search Bar
# -----------------------------
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    user_question = st.text_input(
        "",
        placeholder="search what you want to know",
        label_visibility="collapsed"
    )


# PDF processing
if uploaded_file is not None:

    with pdf.open(uploaded_file) as pdf:

        text = ""

        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n\n"

    st.success("PDF uploaded successfully!")

   #st.write("### 📄 Extracted Text")
    #st.write(text)


    #split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=500,
        chunk_overlap=200
    )
    chunk = text_splitter.split_text(text)
    #st.write(chunk)


    #embedding using openai
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # embedding stored to vector db
    vector_store = FAISS.from_texts(chunk, embeddings)

    #generate answer
    #question -> embeddings -> similarity search -> result to llm -> response (CHAIN ithuku perudhan)

    # usage

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])


    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    #define the LLM and prompts
    llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.3,
    max_output_tokens=1000,
    google_api_key=GOOGLE_API_KEY
    )

    #provide the prompts
    prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant answering questions about a PDF document.\n\n"
     "Guidelines:\n"
     "1. Provide complete, well-explained answers using the context below.\n"
     "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
     "3. If the context mentions related information, include it to give fuller picture.\n"
     "4. Only use information from the provided context - do not use outside knowledge.\n"
     "5. Summarize long information, ideally in bullets where needed\n"
     "6. If the information is not in the context, say so politely.\n"
     "Context:\n{context}"),
    ("human", "{question}")
    ])

    chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    if user_question:
        response = chain.invoke(user_question)
        st.write(response)
