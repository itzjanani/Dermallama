import os
import faiss
import pickle
import numpy as np
import ollama
from pymongo import MongoClient
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# MongoDB setup for storing chat history
client = MongoClient("mongodb://localhost:27017/")
db = client["llama3_chatbot"]
history_collection = db["chat_history"]

# Load the document
DOCUMENT_PATH = "Skin-Disease-Prediction/Oxford-Handbook-of-Medical-Dermatology.txt"  # Change to your document
text_loader = TextLoader(DOCUMENT_PATH)
documents = text_loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Create/Open FAISS Index for RAG
INDEX_PATH = "faiss_index"

if os.path.exists(INDEX_PATH):
    with open(INDEX_PATH, "rb") as f:
        vectorstore = pickle.load(f)
else:
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Use OpenAI embeddings
    vectorstore = FAISS.from_documents(docs, embeddings)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(vectorstore, f)

def retrieve_context(question, top_k=3):
    """Retrieve relevant chunks from FAISS."""
    similar_docs = vectorstore.similarity_search(question, k=top_k)
    return "\n".join([doc.page_content for doc in similar_docs])

def get_chat_history(user_id, num_turns=5):
    """Retrieve past chat history from MongoDB for context."""
    history = history_collection.find({"user_id": user_id}).sort("timestamp", -1).limit(num_turns)
    return "\n".join([f"User: {h['question']}\nAI: {h['response']}" for h in history])

def store_chat_history(user_id, question, response):
    """Save chat history in MongoDB."""
    history_collection.insert_one({"user_id": user_id, "question": question, "response": response})

def ask_llama3(question, user_id="default_user"):
    """
    Main function to interact with the chatbot.
    - Retrieves context from RAG.
    - Uses past chat history.
    - Generates an answer using Ollama.
    """
    retrieved_context = retrieve_context(question)
    chat_history = get_chat_history(user_id)

    prompt = f"""
    You are a helpful AI dermatalogy doctor. Below is the chat history and a new user query.
    
    Chat History:
    {chat_history}

    Relevant Knowledge:
    {retrieved_context}

    User Question: {question}

    Answer the question accurately based on chat history and retrieved knowledge from the dermatology book and rememeber you are profound dermatalogy consultant.
    """

    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"]

    # Store chat history
    store_chat_history(user_id, question, answer)

    return answer

# Example Usage
if __name__ == "__main__":
    print(ask_llama3("What is Ringworms disease"))
