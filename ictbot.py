import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

# Add initial debug message
st.write("Starting application initialization...")

# Load environment variables
load_dotenv(find_dotenv())
st.write("Environment variables loaded")

# Check if HF_TOKEN exists
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    st.error("❌ HF_TOKEN not found in environment variables")
else:
    st.write("✓ HF_TOKEN found")

# Check if vector store path exists
DB_FAISS_PATH = "vectorstore/db_faiss"
if not os.path.exists(DB_FAISS_PATH):
    st.error(f"❌ Vector store path not found: {DB_FAISS_PATH}")
else:
    st.write(f"✓ Vector store path found: {DB_FAISS_PATH}")

@st.cache_resource
def get_vectorstore():
    st.write("Attempting to load vector store...")
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        st.write("✓ Embedding model initialized")
        
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        st.write("✓ Vector store loaded successfully")
        return db
    except Exception as e:
        st.error(f"❌ Failed to load vector store: {str(e)}")
        return None

def set_custom_prompt():
    st.write("Setting up custom prompt...")
    prompt_template = """
    You are Anika, the virtual assistant for ICT Academy of Kerala. You are here to assist users with any questions they may have regarding ICT Academy of Kerala, including information about courses, programs, admissions, events, and other relevant details.
    Use the pieces of information provided in the context to answer the user's question. Always stay within the context of ICT Kerala's website and offerings.

    If you don't know the answer or the information is not available, politely inform the user that you don't know. Avoid making up answers or providing information outside the given context.

    Context: {context}
    Question: {question}

    Provide clear and concise answers. Keep responses professional, friendly, and helpful. No small talk or unrelated details.
    """
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def load_llm():
    st.write("Initializing LLM...")
    try:
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            temperature=0.5,
            model_kwargs={
                "token": hf_token,
                "max_length": 512
            }
        )
        st.write("✓ LLM initialized successfully")
        return llm
    except Exception as e:
        st.error(f"❌ Failed to initialize LLM: {str(e)}")
        return None

def main():
    st.title("Ask Chatbot!")
    st.write("Main function started")

    # Initialize vector store
    vectorstore = get_vectorstore()
    if vectorstore is None:
        st.error("❌ Failed to initialize vector store")
        return

    # Initialize LLM
    llm = load_llm()
    if llm is None:
        st.error("❌ Failed to initialize LLM")
        return

    st.write("✓ All components initialized successfully")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.write("Processing prompt...")
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt()}
            )
            st.write("✓ QA chain created successfully")

            with st.spinner("Thinking..."):
                response = qa_chain.invoke({'query': prompt})
                st.write("✓ Response received")
                
                result = response["result"]
                #source_documents = response["source_documents"]
                result_to_show = f"{result}"

                with st.chat_message('assistant'):
                    st.markdown(result_to_show)
                st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")

if __name__ == "__main__":
    main()