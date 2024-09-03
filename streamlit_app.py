# import streamlit as st
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import UnstructuredPDFLoader
# import os

# # Nastavení API klíče prostřednictvím environmentální proměnné
# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# # Inicializace language modelu s specifikovanými parametry
# llm = ChatOpenAI(temperature=0, max_tokens=1000, model_name="gpt-4-turbo")

# st.title("🤖Kognitivní vyhledávání v obsahu dokumetů.")

# # Informační sekce s příklady použití
# st.info(
#     """
#     Příklady užití:\n
#     -Existuje program speciálně pro hedvábí?\n
#     -Řekni mi vic o programu hedvábí\n
#     ...\n
#     """,
#     icon="🕵️‍♀️",
# )

# # Sidebar pro nahrávání souborů
# with st.sidebar:
#     uploaded_files = st.file_uploader("Výběr souborů PDF", accept_multiple_files=True, type="pdf")

# if uploaded_files:
#     if "processed_data" not in st.session_state:
#         documents = []
#         for uploaded_file in uploaded_files:
#             file_path = os.path.join(os.getcwd(), uploaded_file.name)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.getvalue())

#             loader = UnstructuredPDFLoader(file_path)
#             loaded_documents = loader.load()
#             documents.extend(loaded_documents)

#         text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
#         document_chunks = text_splitter.split_documents(documents)

#         embeddings = OpenAIEmbeddings()
#         vectorstore = Chroma.from_documents(document_chunks, embeddings)

#         st.session_state.processed_data = {
#             "document_chunks": document_chunks,
#             "vectorstore": vectorstore,
#         }
#     else:
#         document_chunks = st.session_state.processed_data["document_chunks"]
#         vectorstore = st.session_state.processed_data["vectorstore"]

#     qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     if prompt := st.chat_input("🤖Co chcete vědět?"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Konsolidace zpráv pro snížení počtu API požadavků
#         chat_history = [(message["role"], message["content"]) for message in st.session_state.messages if message["role"] != "system"]

#         result = qa({"question": prompt, "chat_history": chat_history})

#         source_documents = result.get('source_documents', [])
#         document_attributes = [vars(doc) for doc in source_documents]
#         file_names = [os.path.basename(doc["metadata"]["source"]) for doc in document_attributes if doc["metadata"]["source"] is not None]

#         file_names_string = ', '.join(file_names) if file_names else "Nejsou dostupné žádné soubory"

#         with st.chat_message("assistant"):
#             message_placeholder = st.empty()
#             full_response = result["answer"]
#             message_placeholder.markdown(full_response)
        
#         st.session_state.messages.append({"role": "assistant", "content": full_response})

#         formatted_text = ""
#         if source_documents:
#             page_content = [doc["page_content"] for doc in document_attributes]
#             formatted_text = page_content[0].replace('\n', ' ')

#         with st.expander("Zdrojový text pro odpověď"):
#             st.write(formatted_text)
# else:
#     st.write("Prosím nahrajte soubory PDF.")


from openai import OpenAI
import streamlit as st

st.title("Oficiálně podporovaný a plně v souladu se všemi předpisy: GPT klon, který nepřekračuje žádné stanovy ani pravidla Zscaleru")

client = OpenAI(api_key=st.secrets["sk-proj-KF8TKKi5FVeFfvd1g0ssT3BlbkFJnKESAChVOL5fMC5CdrXY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
