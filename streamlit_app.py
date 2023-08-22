import streamlit as st
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredPDFLoader
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo")

st.title("🤖MarFin 0.0.27\n (Kognitivní vyhledávání v obsahu dokumetů. Jako by to někdo potřeboval.)")

with st.sidebar:
    uploaded_files = st.file_uploader("Výběr souborů PDF", accept_multiple_files=True, type="pdf")

if uploaded_files:
    print(f"Počet nahraných souborů: {len(uploaded_files)}")

    if "processed_data" not in st.session_state:
        documents = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(os.getcwd(), uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = UnstructuredPDFLoader(file_path)
            loaded_documents = loader.load()
            print(f"Počet načtených souborů: {len(loaded_documents)}")

            documents.extend(loaded_documents)

        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        document_chunks = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(document_chunks, embeddings)

        st.session_state.processed_data = {
            "document_chunks": document_chunks,
            "vectorstore": vectorstore,
        }

        print(f"Celkový počet částí: {len(document_chunks)}")

    else:
        document_chunks = st.session_state.processed_data["document_chunks"]
        vectorstore = st.session_state.processed_data["vectorstore"]

    qa = load_qa_with_sources_chain(llm, chain_type='stuff')  # Nahrazení původního řetězce

    # Zbytek kódu zůstává stejný
