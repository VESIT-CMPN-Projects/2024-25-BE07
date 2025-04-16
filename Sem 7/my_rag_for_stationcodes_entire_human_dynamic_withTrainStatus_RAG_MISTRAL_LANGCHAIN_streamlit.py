import streamlit as st
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Set the API keys
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")
os.environ["HUGGING_FACE_API"] = os.getenv("HF_API_KEY")

# Streamlit page configuration
st.set_page_config(page_title="Indian Railways RAG System", layout="centered")

# Streamlit UI components
st.title("Konkan Railways Query System")
st.write("Enter your query below to get information about Indian Railways.")

# Step 1: Load and Extract Text from JSON Files
@st.cache_data(show_spinner=False)
def extract_text_and_trains_from_json():
    json_station_file_path = 'konkanRailwayCodeNameZone.json'
    json_train_status_file_path = 'train_status1.json'
    
    # Extract station data
    with open(json_station_file_path, 'r') as file:
        station_data = json.load(file)
    
    station_docs = [
        f"Station Code: {entry['CODE']}, Station Name: {entry['STATION NAME']}, Railway Zone: {entry['RAILWAY ZONE']}"
        for entry in station_data
    ]
    
    # Extract train status data
    with open(json_train_status_file_path, 'r') as file:
        train_status_data = json.load(file)
    
    train_docs = []
    train_names = set()  # Use a set to ensure unique train names
    for timestamp, trains in train_status_data.items():
        for train in trains:
            document_text = (f"Timestamp: {timestamp}, Train No: {train['Train No']}, Train Name: {train['Train Name']}, "
                             f"Status: {train['Status']}, Station: {train['Station']}, Time: {train['Time']}, "
                             f"Delay: {train['Delay']}")
            train_docs.append(document_text)
            train_names.add(train['Train Name'])  # Add train name to the set
    
    return train_docs + station_docs, sorted(train_names)  # Combine docs and sort train names

# Load documents and train names
if "documents" not in st.session_state or "train_names" not in st.session_state:
    st.session_state.documents, st.session_state.train_names = extract_text_and_trains_from_json()

documents = st.session_state.documents
train_names = st.session_state.train_names

# Step 2: Load Model and Index
@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(documents)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    return model, index

model, index = load_model_and_index()

# Step 3: Retrieve Relevant Documents Based on User Query
def retrieve_documents(query, max_k=10, threshold=0.8):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), max_k)

    retrieved_docs = [
        Document(page_content=documents[idx])
        for dist, idx in zip(D[0], I[0])
        if dist <= threshold
    ]
    return retrieved_docs

# Step 4: Mistral LLM Integration and QA Chain Setup
@st.cache_resource
def setup_llm_chain():
    prompt_template = PromptTemplate.from_template(
        """
        You are a highly knowledgeable assistant with access to detailed information about Indian Railways. 
        Your task is to answer the following question based on the retrieved relevant documents. 
        Be specific, concise, and focus on the most relevant information. Prioritize data from the JSON files. 
        The documents provided to you have the current status using the Timestamp. If there is any query that says "current status" or "train status" convert it to "current train status" internally and give the appropriate answer.

        Relevant documents: {context}

        Question: {question}

        Answer:
        """
    )
    llm = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.2,
    )
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)

qa_chain = setup_llm_chain()

def generate_llm_response(query, retrieved_docs):
    response = qa_chain.invoke({"input_documents": retrieved_docs, "question": query})
    return response['output_text']

# Dynamic input for query
user_query = st.text_input("Your Query", placeholder="e.g., What is the current train status of Train ABC")

# Suggest train names dynamically if query starts with the prefix
if user_query.startswith("What is the current train status of"):
    suggested_train = st.selectbox(
        "Choose a train name to complete your query:",
        options=[""] + train_names,
        help="Select from the available train names."
    )
    if suggested_train:
        user_query = f"What is the current train status of {suggested_train}"


# Generate response if the user provides a query
if user_query:
    with st.spinner("Retrieving documents and generating response..."):
        retrieved_docs = retrieve_documents(user_query, max_k=10, threshold=0.8)
        
        response = generate_llm_response(user_query, retrieved_docs)
        st.success("Response:")
        st.write(response)

# Place "Refresh Documents" and "Not what you are looking for?" buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("Refresh Documents"):
        with st.spinner("Refreshing documents..."):
            # Reload the documents and train names
            new_documents, new_train_names = extract_text_and_trains_from_json()
            
            # Update the session state
            st.session_state.documents = new_documents
            st.session_state.train_names = new_train_names
            
            # Provide feedback
            st.success("Documents refreshed successfully!")

with col2:
    if st.button("Not what you are looking for?"):
        st.warning("Feel free to refine your query or try asking another question.")


# Use the updated documents and train names
documents = st.session_state.documents
train_names = st.session_state.train_names