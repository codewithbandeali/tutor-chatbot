from flask import Flask, render_template, request, jsonify
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import re
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os
from werkzeug.utils import secure_filename
from prompt import *  # Make sure you have the necessary prompts in 'prompt.py'
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tempfile import NamedTemporaryFile

from flask_cors import CORS
UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)
CORS(app)  # Enables CORS for all routes
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'txt', 'pptx'}
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Function to check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to process uploaded file
def process_file(uploaded_file):
    if uploaded_file and allowed_file(uploaded_file.filename):
        with NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.filename.split('.')[-1]}") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        file_extension = uploaded_file.filename.split('.')[-1].lower()
        loader = None
        if file_extension == "pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == "docx":
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == "txt":
            loader = TextLoader(temp_file_path)
        elif file_extension == "pptx":
            loader = UnstructuredPowerPointLoader(temp_file_path)

        if loader:
            documents = loader.load()
            os.remove(temp_file_path)  # Clean up temporary file
            return documents

    return None

# Split documents into smaller chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20, separators=["\n\n", "\n", " ", ".", ","])
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Store embeddings
def embeddings_store(docs, persist_directory='data/chroma_rag'):
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    
    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    )
    vector_langchain_chroma = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    return vector_langchain_chroma

# Load embeddings
def load_embeddings(persist_directory='data/chroma_rag'):
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",

    )
    load_langchain_chroma = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return load_langchain_chroma
def clean_and_format_text(text):
    # Remove unwanted ** symbols (if any)
    text = re.sub(r'\*\*', '', text)
    
    # Ensure numbered questions (e.g., 1., 2.) start on new lines
    text = re.sub(r'(\d+\.\s+)', r'\n\1', text)  # Ensures 1., 2., etc., are on new lines
    
    # Ensure multiple choice options (e.g., a), b), etc.) start on new lines
    text = re.sub(r'([a-d]\)\s+)', r'\n\1', text)  # Ensures a), b), etc., are on new lines

    # Ensure new lines after headers (like "Instructions:" or "Answers:")
    text = re.sub(r'(###.*?|Instructions:|Answers:)', r'\n\1\n', text)

    # Remove extra whitespaces or unnecessary newlines
    text = re.sub(r'\s*\n\s*\n+', '\n\n', text)

    return text

# Load embeddings once at startup
embeddings = load_embeddings()
retriever = embeddings.as_retriever(search_kwargs={'k': 2})

# Define the prompt templates
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", retriever_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, openai_api_key=OPENAI_API_KEY)
history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/upload_file", methods=["POST","GET"])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file:
        documents = process_file(uploaded_file)
        if documents:
            text_chunks = text_split(documents)
            embeddings_store(text_chunks)
            return "File uploaded and processed successfully"
    return "File upload failed", 400

@app.route("/get", methods=["POST"])
def chat():
    chat_history = []
    query = request.form["msg"]
    
    # Invoke the RAG chain to get the result
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    
    # Extract only the answer portion
    answer = result.get('answer', 'No answer available')
    
    # Clean and format the answer
    #formatted_answer = clean_and_format_text(answer)
    
    return answer

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
