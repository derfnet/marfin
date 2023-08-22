import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredPDFLoader
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo")

st.title("🤖MarFin 0.0.26\n (Kognitivní vyhledávání v obsahu dokumetů. Jako by to někdo potřeboval.)")

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

    qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("🤖Co chcete vědět? I když pochybuji, že vám mohu pomoci."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})

        for source in result:
            print(source)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            full_response = result["answer"]
            message_placeholder.markdown(full_response + "|")
        message_placeholder.markdown(full_response)    
        print(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.write("Prosím nahrajte soubory PDF.")
