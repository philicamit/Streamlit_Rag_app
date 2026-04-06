import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
from supabase.client import Client, create_client
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- 1. CONFIG & CONNECTIONS ---
st.set_page_config(page_title="Corporate Policy AI", layout="wide")
st.title("📄 Corporate Policy RAG Assistant")

# Initialize Clients
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(url, key)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 2. DOCUMENT PROCESSING LOGIC ---
def process_and_upload(uploaded_file):
    # Save to a temporary file to load it
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # Select loader
        if uploaded_file.name.endswith('.docx'):
            loader = Docx2txtLoader(tmp_path)
        else:
            loader = PyPDFLoader(tmp_path)
        
        docs = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        
        # Create vectors and prepare for Supabase
        vectors_to_insert = []
        for chunk in chunks:
            vector = embeddings.embed_query(chunk.page_content)
            vectors_to_insert.append({
                "content": chunk.page_content,
                "embedding": vector,
                "metadata": {"source": uploaded_file.name}
            })
        
        # Push to Supabase
        supabase.table("documents").insert(vectors_to_insert).execute()
        return len(chunks)
    finally:
        os.remove(tmp_path)

# --- 3. RETRIEVAL LOGIC ---
def custom_retriever(query_text):
    query_vector = embeddings.embed_query(query_text)
    
    response = supabase.rpc("match_documents", {
        "query_embedding": query_vector,
        "match_threshold": 0.2,
        "match_count": 5
    }).execute()
    
    if not response.data:
        return None
    
    return "\n\n".join([f"Source: {res['metadata'].get('source', 'Unknown')}\n{res['content']}" for res in response.data])

# --- 4. SIDEBAR: UPLOAD SECTION ---
with st.sidebar:
    st.header("📤 Upload Documents")
    new_files = st.file_uploader("Upload PDF or Word files", type=["pdf", "docx"], accept_multiple_files=True)
    
    if st.button("Index Documents"):
        if new_files:
            with st.spinner("Processing..."):
                count = 0
                for f in new_files:
                    count += process_and_upload(f)
                st.success(f"Successfully indexed {count} chunks!")
        else:
            st.warning("Please select files first.")

# --- 5. MAIN CHAT INTERFACE ---
st.subheader("💬 Ask a Policy Question")
query = st.text_input("What would you like to know?", placeholder="e.g., What is the sick leave policy?")

if query:
    with st.spinner("Searching..."):
        context_text = custom_retriever(query)
        
        if not context_text:
            st.error("No relevant information found in the database.")
        else:
            prompt = ChatPromptTemplate.from_template(
                "Answer based only on the context provided. If not mentioned, say you don't know.\n\nContext:\n{context}\n\nQuestion: {input}"
            )
            final_prompt = prompt.format(context=context_text, input=query)
            result = llm.invoke(final_prompt)
            
            st.markdown("### Answer")
            st.info(result.content)