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

# Step 1: Load and Extract Text from JSON Files
def extract_text_from_station_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    documents = []
    for entry in data:
        document_text = f"Station Code: {entry['CODE']}, Station Name: {entry['STATION NAME']}, Railway Zone: {entry['RAILWAY ZONE']}"
        documents.append(document_text)
    
    return documents

def extract_text_from_train_status_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    documents = []
    for timestamp, trains in data.items():
        for train in trains:
            document_text = (f"Timestamp: {timestamp}, Train No: {train['Train No']}, Train Name: {train['Train Name']}, "
                             f"Status: {train['Status']}, Station: {train['Station']}, Time: {train['Time']}, "
                             f"Delay: {train['Delay']}")
            documents.append(document_text)
    
    return documents

# Load both JSON files (replace with your file paths)
json_station_file_path = 'konkanRailwayCodeNameZone.json'
json_train_status_file_path = 'train_status1.json'

# Extract documents from both JSON files
station_documents = extract_text_from_station_json(json_station_file_path)
train_status_documents = extract_text_from_train_status_json(json_train_status_file_path)

# Combine both sets of documents
documents = train_status_documents  + station_documents 
print("Documents extracted:", documents[:5])  # Print the first few documents

# Step 2: Convert Documents to Embeddings
# Using sentence-transformers for generating document embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings for the extracted documents
embeddings = model.encode(documents)
print("Embeddings shape:", embeddings.shape)

# Step 3: Store Embeddings in FAISS Vector Store
dimension = embeddings.shape[1]  # Number of dimensions in the embeddings
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the FAISS index
index.add(np.array(embeddings))
print("Embeddings added to FAISS index.")

# Step 4: Retrieve Relevant Documents Based on User Query
def retrieve_documents(query, max_k=10, threshold=0.8):
    # Convert the query into an embedding
    query_embedding = model.encode([query])
    print("Query embedding:", query_embedding)
    
    # Search for top-k similar documents
    D, I = index.search(np.array(query_embedding), max_k)  # Retrieves up to max_k documents
    print("Distances:", D)
    print("Indices:", I)
    
    # Filter results based on the distance threshold
    retrieved_docs = []
    for dist, idx in zip(D[0], I[0]):
        if dist <= threshold:  # Only include documents that meet the similarity threshold
            retrieved_docs.append(Document(page_content=documents[idx]))  # Wrap in Document format
    
    return retrieved_docs

# **Step 5: Mistral LLM Integration with Prompt Template**

# Create a custom prompt template for the Mistral LLM
prompt_template = PromptTemplate.from_template(
    """
    You are a highly knowledgeable assistant with access to detailed information about Indian Railways. Your task is to answer the following question based on the retrieved relevant documents. Be specific, concise, and focus on the most relevant information. Prioritize data from the JSON files. The documents provided to you have the current status using the Timestamp. Be smart and understand the documents provided to you.
    Lastly, if I ask a question in hinglish/minglish, I want the answer in similar fashion. Also, answer in the same language that the user asks in.

    Relevant documents: {context}

    Question: {question}

    Answer:
    """
)

# Step 6: Mistral LLM and QA Chain Setup
llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.2,
)

qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)

def generate_llm_response(query, retrieved_docs):
    # Use `invoke` instead of `run`
    response = qa_chain.invoke({"input_documents": retrieved_docs, "question": query})
    return response

# Step 7: Run the RAG Pipeline
def run_rag_pipeline(query):
    # Retrieve relevant documents based on query
    retrieved_docs = retrieve_documents(query, max_k=10, threshold=0.8)
    print("Retrieved documents:", retrieved_docs)

    # # Generate the response using Mistral LLM
    # if not retrieved_docs:
    #     return "I'm not sure about that. Let me look into it further and get back to you!"

    response = generate_llm_response(query, retrieved_docs)
    
    return response['output_text']  # Extract and return the output text

# Step 8: Example Run (Querying and Generating a Response)
user_query = input("Enter your query: ")
response = run_rag_pipeline(user_query)
print("Generated Response:\n", response)
