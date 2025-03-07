import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory 


## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm

# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
You are Arc, the virtual assistant for ICT Academy of Kerala. You are here to assist users with any questions they may have regarding ICT Academy of Kerala, including information about courses, programs, admissions, events, and other relevant details.
Use the pieces of information provided in the context to answer the user's question. Always stay within the context of ICT Kerala's website and offerings.

If you don't know the answer or the information is not available, politely inform the user that you don't know. Avoid making up answers or providing information outside the given context.

Context: {context}
Question: {question}

Provide clear and concise answers. Keep responses professional, friendly, and helpful. No small talk or unrelated details.
"""


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

memory = ConversationBufferMemory(memory_key="chat_history", input_key="query", output_key="result")


# Create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
    memory=memory  # Attach memory to the chain
)

# Step 6: Interaction with user
while True:
    user_query = input("Write Query Here: ")
    
    # Now invoke with the query and include memory
    response = qa_chain.invoke({'query': user_query})
    
    # Print the result and source documents
    print("RESULT: ", response["result"])
    print("SOURCE DOCUMENTS: ", response["source_documents"])