import os
import io
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, jsonify
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchFieldDataType, VectorSearch, VectorSearchAlgorithmConfiguration
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from dotenv import load_dotenv
import PyPDF2

# Loading all the environment variables containing the API keys, endpoints, and other crucial connection strings
load_dotenv()

# Initializing the Flask application
app = Flask(__name__)

# Azure Blob Storage
# Getting the blob storage connection string
blob_service_client = BlobServiceClient.from_connection_string(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))
container_name = os.getenv('AZURE_STORAGE_CONTAINER_NAME')
# We need to be able to add data to our container, and the previous line lets us add containers themselves, as it interacts with blob storage on an accout level
container_client = blob_service_client.get_container_client(container_name)

# Azure AI Search
# Getting the search instance
search_client = SearchClient(
    endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
    index_name=os.getenv('AZURE_SEARCH_INDEX_NAME'),
    credential=AzureKeyCredential(os.getenv('AZURE_SEARCH_KEY'))
)
# Getting the index used to embed the raw data
index_client = SearchIndexClient(
    endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
    credential=AzureKeyCredential(os.getenv('AZURE_SEARCH_KEY'))
)

# Azure OpenAI
# Getting the OpenAI instance for the chat models (notice we don't use the embedding models, as it happens in the background through the AI Search indexing process)
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-08-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def create_search_index():
    index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="content", type=SearchFieldDataType.String, searchable=True),
        SimpleField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_configuration="vector_config")
    ]
    vector_search = VectorSearch(
        algorithm_configurations=[
            VectorSearchAlgorithmConfiguration(
                name="vector_config",
                kind="hnsw"
            )
        ]
    )
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search
    )
    index_client.create_index(index)

# Uncomment the following line and run the script once to create the index
# create_search_index()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # To handle the file upload
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Save the file to blob storage
            blob_client = container_client.get_blob_client(uploaded_file.filename)
            blob_client.upload_blob(uploaded_file.read(), overwrite=True)
            # Index the document
            index_document(uploaded_file.filename)
        return redirect(url_for('index'))
    else:
        # List blobs in the container
        blob_list = container_client.list_blobs()
        files = [blob.name for blob in blob_list]
        return render_template('index.html', files=files)
    
@app.route('/view/<filename>')
def view_document(filename):
    # Generate a SAS URL for the blob
    sas_token = generate_blob_sas(
        account_name=blob_service_client.account_name,
        container_name=container_name,
        blob_name=filename,
        account_key=blob_service_client.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(datetime.timezone.utc) + timedelta(hours=1)  
    )
    document_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{filename}?{sas_token}"
    # List blobs in the container
    blob_list = container_client.list_blobs()
    files = [blob.name for blob in blob_list]
    return render_template('index.html', files=files, document_url=document_url)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']

    # Generate embeddings for the user query
    embedding_response = client.embeddings.create(
        input=user_input,
        model='text-embedding-ada-002'
    )
    query_vector = embedding_response.data[0].embedding

    # Searching for relevant documents now, with the query vector
    search_results = search_client.search(
        search_text="",
        vector_queries=[{
            "value":query_vector,
            "fields":"text_vector",
            "k":4
        }],
        select=['chunk']
    )
    # Compiling the context from the selected search results
    context = "\n".join([doc['chunk'] for doc in search_results])

    # Generate an LLM response using the context for the query provided and embedded
    messages = [
        {"role": "system", "content":"You are an expert in navigating your way through documents. Please help the users in the best way you can. Be concise, and short with your responses."},
        {"role": "assistant", "content": context},
        {"role": "user", "content": user_input}
    ]
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=messages,
        temperature=0.8,
        max_tokens=540
    )

    assistant_reply = response['choices'][0]['message']['content']

    return jsonify({'reply': assistant_reply})

def index_document(file_name):
    # Get the file from azure blob storage
    blob_client = container_client.get_blob_client(file_name)
    download_stream = blob_client.download_blob()
    pdf_bytes = download_stream.readall()

    # Extract text from PDF
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Generate embeddings for the document content
    embedding_response = client.embeddings.create(
        input=text,
        model='text-embedding-ada-002'
    )
    doc_embeddings = embedding_response.data[0].embedding

    doc = {
        "id":file_name,
        "content":text,
        "embedding":doc_embeddings
    }

    # Upload the document to the search index
    search_client.upload_documents(documents=[doc])