import os
from dotenv import load_dotenv
from supabase.client import Client, create_client
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- 1. CONNECT ---
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY") 
supabase: Client = create_client(url, key)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- 2. INITIALIZE VECTOR STORE ---
# This points to the table and the function you just created
vector_db = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents",
)

# --- 3. STABLE RETRIEVAL BYPASS ---
def custom_retriever(query_text):
    # 1. Turn your question into a vector
    query_vector = embeddings.embed_query(query_text)
    
    # 2. Call the RPC function
    response = supabase.rpc("match_documents", {
        "query_embedding": query_vector,
        "match_threshold": 0.2, # Lowered slightly to ensure we get results
        "match_count": 5
    }).execute()
    
    # 3. Check if we actually got data back
    if not response.data:
        return "No relevant documents found in the database."
    
    # 4. Format the results
    return "\n\n".join([res['content'] for res in response.data])

# --- 4. RUN THE CHAIN ---
query = "What is the policy regarding leave and attendance?"
print("🔍 Searching your 15 documents (Manual Mode)...")

context_text = custom_retriever(query)

# Check context before sending to LLM
if "No relevant documents" in context_text:
    print("⚠️ Warning: No matching chunks found. Check if your 'documents' table has data.")
else:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question based only on the provided context:\n\n{context}\n\nQuestion: {input}"
    )

    final_prompt = prompt.format(context=context_text, input=query)
    result = llm.invoke(final_prompt)
    print(f"\nAssistant: {result.content}")