# 📄 Corporate Policy AI: Multi-Document RAG Assistant

A professional-grade **Retrieval-Augmented Generation (RAG)** pipeline designed to automate HR and Policy inquiries. This application allows users to upload corporate documents (PDF/Docx), indexes them into a vector database, and provides precise, context-aware answers using LLMs.

---

## 🚀 Key Features
* **Multi-Format Support:** Ingests and processes both `.pdf` and `.docx` files.
* **Vector Search:** Utilizes **OpenAI Embeddings** and **Supabase (PostgreSQL/pgvector)** for high-performance similarity searching.
* **Streamlit UI:** A clean, user-friendly web interface for document management and real-time chatting.
* **Contextual Accuracy:** Uses a custom PostgreSQL RPC function (`match_documents`) to ensure the LLM only answers based on provided corporate data.

---

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Framework:** LangChain
* **Frontend:** Streamlit
* **Database:** Supabase (PostgreSQL + pgvector)
* **AI Models:** * `text-embedding-3-small` (Embeddings)
    * `gpt-4o-mini` (Reasoning & Generation)

---

## 🏗️ Architecture
1.  **Ingestion:** Documents are loaded and split into 1000-character chunks using `RecursiveCharacterTextSplitter`.
2.  **Vectorization:** Chunks are converted into 1536-dimensional vectors.
3.  **Storage:** Vectors and metadata (source filename) are stored in Supabase.
4.  **Retrieval:** User queries are vectorized and compared against the database using **Cosine Similarity**.
5.  **Generation:** The top 5 relevant chunks are sent to GPT-4o-mini as context to generate a final answer.

---

## ⚙️ Local Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/philicamit/Streamlit_Rag_app.git](https://github.com/philicamit/Streamlit_Rag_app.git)
    cd Streamlit_Rag_app
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables:**
    Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY=your_openai_key
    SUPABASE_URL=your_supabase_url
    SUPABASE_SERVICE_KEY=your_service_role_key
    ```

4.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

---

## 📊 Author
**Amit Rastogi** 