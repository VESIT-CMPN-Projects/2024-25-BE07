import streamlit as st
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Set the API keys
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")
os.environ["HUGGING_FACE_API"] = os.getenv("HF_API_KEY")

# Step 1: Load and Extract Text from JSON Files
def extract_text_and_trains_from_json():
    json_station_file_path = 'konkanRailwayCodeNameZone.json'
    json_train_status_file_path = 'train_status2.json'
    
    with open(json_station_file_path, 'r') as file:
        station_data = json.load(file)
    station_docs = [
        f"Station Code: {entry['CODE']}, Station Name: {entry['STATION NAME']}, Railway Zone: {entry['RAILWAY ZONE']}"
        for entry in station_data
    ]
    
    with open(json_train_status_file_path, 'r') as file:
        train_status_data = json.load(file)
    
    train_docs = []
    train_names = set()
    for timestamp, trains in train_status_data.items():
        for train in trains:
            document_text = (f"Timestamp: {timestamp}, Train No: {train['Train No']}, Train Name: {train['Train Name']}, "
                             f"Status: {train['Status']}, Station: {train['Station']}, Time: {train['Time']}, "
                             f"Delay: {train['Delay']}")
            train_docs.append(document_text)
            train_names.add(train['Train Name'])
    
    return train_docs + station_docs, sorted(train_names)

# Step 2: Load Model and Index
def load_model_and_index(documents):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(documents)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    return model, index

# Step 3: Retrieve Relevant Documents
def retrieve_documents(query, max_k=10, threshold=0.8):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), max_k)
    
    retrieved_docs = [
        Document(page_content=documents[idx])
        for dist, idx in zip(D[0], I[0])
        if dist <= threshold
    ]
    return retrieved_docs

# Step 4: LLM, Memory, and QA Chain Setup
@st.cache_resource
def setup_llm_chain():
    prompt_template = PromptTemplate.from_template(
        """
        You are a highly knowledgeable assistant with access to detailed information about Indian Railways. Your task is to answer the following question based on the retrieved relevant documents. Be specific, concise, and focus on the most relevant information. Prioritize data from the JSON files. The documents provided to you have the current status using the Timestamp. Be smart and understand the documents provided to you.
        Lastly, if I ask a question in hinglish/minglish, I want the answer in similar fashion. Also, answer in the same language that the user asks in.
        
        Conversation history:
        {history}
        
        Relevant documents: {context}
        
        Question: {question}
        
        Answer:
        """
    )
    llm = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.1,
    )
    memory = ConversationBufferMemory(memory_key="history", input_key="question")
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt_template, memory=memory)

# Streamlit App Configuration
st.set_page_config(page_title="Indian Railways RAG System", layout="wide")

# Initialize session state variables
if "documents" not in st.session_state:
    st.session_state.documents, st.session_state.train_names = extract_text_and_trains_from_json()

if "model" not in st.session_state or "index" not in st.session_state:
    st.session_state.model, st.session_state.index = load_model_and_index(st.session_state.documents)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Assign to local variables
documents = st.session_state.documents
train_names = st.session_state.train_names
model, index = st.session_state.model, st.session_state.index
qa_chain = setup_llm_chain()

# Sidebar with options and help
with st.sidebar:
    st.title("🔧 Options")
    if st.button("🔄 Refresh Data"):
        st.session_state.documents, st.session_state.train_names = extract_text_and_trains_from_json()
        st.session_state.model, st.session_state.index = load_model_and_index(st.session_state.documents)
        st.rerun()

    st.markdown("""
    **Help**
    - Ask about train statuses and station details.
    - Use precise train names or station codes for better results.
    - Click Refresh if the data seems outdated.
    """)

# Main App Interface
st.title("🚆 Konkan Railways Query System")
st.write("Ask about train statuses, station details, and more!")

# Chat Interface
st.write("""
<style>
    .chat-container {
        display: flex;
        flex-direction: column;
        max-height: 500px;
        overflow-y: auto;
        padding-bottom: 10px;
    }
    .chat-bubble {
        background-color: #f3f3f3;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        max-width: 80%;
    }
    .user {
        align-self: flex-end;
        background-color: #0084FF;
        color: white;
    }
    .bot {
        align-self: flex-start;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for query, response in st.session_state.chat_history:
    st.markdown(f'<div class="chat-bubble user"><b>You:</b> {query}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-bubble bot"><b>Bot:</b> {response}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input Box
user_query = st.text_input("Your Query", placeholder="e.g., What is the current train status of Train ABC")
submit_button = st.button("↗️ Submit")

def generate_llm_response(query, retrieved_docs):
    response = qa_chain.invoke({"question": query, "context": retrieved_docs, "input_documents": retrieved_docs})
    return response['output_text']

if submit_button and user_query:
    with st.spinner("Retrieving and generating response..."):
        retrieved_docs = retrieve_documents(user_query, max_k=10, threshold=0.8)
        response = generate_llm_response(user_query, retrieved_docs)
        
        st.session_state.chat_history.append((user_query, response))
        st.rerun()