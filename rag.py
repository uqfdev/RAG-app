import streamlit as st
from PyPDF2 import PdfReader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os

load_dotenv(override=True)

prompt_template = """
Answer the following question based only on provided context The context is about NOTE DE
CONJONCTURE DIRECTION DES ETUDES ET DES PREVISIONS FINANCIERES Décembre 2025
The context is delimited by <context> tag
The user question is delimited by <question> tag
If the answer is not found in the context, answer: Je ne sais pas!
<context>
{context}
</context>
<question>
{question}
</question>
"""

llm = ChatOllama(model="nemotron-3-super:cloud", temperature=0)


def init_session_state():
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None


def build_retriever(pdf_docs):
    content = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                content += text + "\n"

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name='o200k_base',
        chunk_size=300,
        chunk_overlap=20
    )
    chunks = splitter.split_text(content)

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma.from_texts(
        chunks,
        embedding_model,
        collection_name="pdf_collection",
    )
    return vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})


def rag_query(query, retriever, llm, prompt_template):
    """RAG function as defined in your notebook"""
    context_docs = retriever.invoke(query)
    context_list = [d.page_content for d in context_docs]
    context_for_query = ". ".join(context_list)
    prompt = prompt_template.format(context=context_for_query, question=query)
    resp = llm.invoke(prompt)
    return resp.content


def main():
    st.set_page_config(page_title="RAG Application", layout="wide")
    init_session_state()

    st.title("Retrieval Augmented Generation")
    st.markdown("---")

    with st.sidebar:
        st.header("Data Loader")
        
        st.image("upm_rag.png")  
        pdf_docs = st.file_uploader(
            "Load your PDFs", 
            accept_multiple_files=True, 
            type=['pdf'],
            help="Upload one or more PDF files"
        )

        if st.button("Submit and Build Retriever", type="primary"):
            if not pdf_docs:
                st.warning("⚠️ Please upload at least one PDF file before submitting.")
            else:
                with st.spinner("Loading PDFs and building embeddings..."):
                    try:
                        st.session_state.retriever = build_retriever(pdf_docs)
                        st.success("PDFs loaded and retriever created successfully!")
                    except Exception as e:
                        st.error(f"Error building retriever: {str(e)}")

    st.header("Chatbot")
    user_question = st.text_input("Ask your question:", placeholder="e.g., Qu'il est la capitalisation boursière de Maroc pour 2025?")

    if user_question:
        if not st.session_state.retriever:
            st.error("Please upload PDFs and click 'Submit' before asking a question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    answer = rag_query(
                        user_question, 
                        st.session_state.retriever, 
                        llm, 
                        prompt_template
                    )
                    
                    st.markdown("### Answer:")
                    st.write(answer)
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")


    st.markdown("---")
    st.caption("Powered by Ollama | Model: nemotron-3-super:cloud | Embeddings: nomic-embed-text")


if __name__ == "__main__":
    main()