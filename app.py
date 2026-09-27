# ============================================================
# NEXA — DDS ENTERPRISE HR ASSISTANT
# Gemini + LlamaIndex + Pinecone + Gradio
# ============================================================

import os
from dotenv import load_dotenv
from pathlib import Path

import gradio as gr
from pinecone import Pinecone

from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Use the Gemini model that is working for your API key.
GEMINI_MODEL = "gemini-3.5-flash-lite"

EMBEDDING_MODEL = "gemini-embedding-001"

PINECONE_INDEX_NAME = "nexa-gemini-hr"
PINECONE_NAMESPACE = "nexa-hr-v1"

PDF_FOLDER = Path("DDS_HR_Handbook")


# ============================================================
# 2. CHECK API KEYS
# ============================================================

print("=" * 60)
print("NEXA — DDS ENTERPRISE HR ASSISTANT")
print("=" * 60)

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing.\n"
        "Set it as an environment variable before running app.py."
    )

if not PINECONE_API_KEY:
    raise RuntimeError(
        "PINECONE_API_KEY is missing.\n"
        "Set it as an environment variable before running app.py."
    )

print("✅ Gemini API key found")
print("✅ Pinecone API key found")


# ============================================================
# 3. CONNECT GEMINI EMBEDDINGS
# ============================================================

print("\nConnecting Gemini embeddings...")

Settings.embed_model = GoogleGenAIEmbedding(
    model_name=EMBEDDING_MODEL,
    api_key=GEMINI_API_KEY
)

print("✅ Gemini embeddings connected")


# ============================================================
# 4. CONNECT GEMINI LLM
# ============================================================

print("\nConnecting Gemini LLM...")

Settings.llm = GoogleGenAI(
    model=GEMINI_MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0.2
)

print(f"✅ Gemini LLM connected: {GEMINI_MODEL}")


# ============================================================
# 5. CONNECT PINECONE
# ============================================================

print("\nConnecting Pinecone...")

pc = Pinecone(api_key=PINECONE_API_KEY)

available_indexes = pc.list_indexes().names()

print("Available indexes:", available_indexes)

if PINECONE_INDEX_NAME not in available_indexes:
    raise RuntimeError(
        f"Pinecone index '{PINECONE_INDEX_NAME}' was not found.\n"
        f"Available indexes: {available_indexes}"
    )

pinecone_index = pc.Index(PINECONE_INDEX_NAME)

index_info = pc.describe_index(PINECONE_INDEX_NAME)

print("✅ Connected to Pinecone")
print("Index:", PINECONE_INDEX_NAME)
print("Dimension:", index_info["dimension"])
print("Metric:", index_info["metric"])


# ============================================================
# 6. CONNECT LLAMAINDEX TO PINECONE
# ============================================================

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    namespace=PINECONE_NAMESPACE
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

print("✅ LlamaIndex connected to Pinecone")


# ============================================================
# 7. LOAD HR PDFs
# ============================================================

print("\nChecking HR documents...")

if not PDF_FOLDER.exists():
    raise RuntimeError(
        f"PDF folder not found:\n{PDF_FOLDER.resolve()}\n\n"
        "Make sure your project looks like:\n"
        "HR-Case Navigator/\n"
        "├── app.py\n"
        "└── DDS_HR_Handbook/\n"
        "    ├── DDS sample.pdf\n"
        "    ├── DDS sample_2.pdf\n"
        "    ├── DDS sample_3.pdf\n"
        "    └── DDS sample_4.pdf"
    )

pdf_files = list(PDF_FOLDER.glob("*.pdf"))

if not pdf_files:
    raise RuntimeError(
        f"No PDF files found inside {PDF_FOLDER.resolve()}"
    )

print(f"Found {len(pdf_files)} HR PDF files:")

for pdf in pdf_files:
    print(" -", pdf.name)


# ============================================================
# 8. LOAD DOCUMENTS
# ============================================================

print("\nLoading HR documents...")

documents = SimpleDirectoryReader(
    input_dir=str(PDF_FOLDER),
    required_exts=[".pdf"],
    file_extractor={
        ".pdf": PDFReader()
    }
).load_data()

print(f"✅ Loaded {len(documents)} document sections")


# ============================================================
# 9. CONNECT TO EXISTING VECTOR INDEX
# ============================================================

print("\nConnecting Nexa to the HR knowledge base...")

hr_index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context
)

print("✅ Nexa knowledge base connected")


# ============================================================
# 10. NEXA SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_PROMPT = """
You are Nexa, the DDS Enterprise HR Assistant.

Your authorized knowledge source is the approved DDS HR policy
documents retrieved for the current question.

Rules:

1. Answer only using information supported by the retrieved HR documents.

2. Never invent, guess, or use outside knowledge.

3. Do not provide confidential employee information.

4. Do not provide salary information.

5. If the retrieved documents do not contain enough information,
   clearly say that the information is unavailable.

6. If the question is outside DDS HR policy scope,
   politely explain that Nexa can only assist with DDS HR policy questions.

7. If the question is unclear, ask the user to clarify.

8. Keep answers concise, professional, and easy to understand.

9. Never reveal chain-of-thought or private reasoning.

10. Do not add information that is not supported by the retrieved documents.

11. When possible, explain the relevant policy clearly and directly.
"""


# ============================================================
# 11. CREATE QUERY ENGINE
# ============================================================

print("\nCreating Nexa RAG query engine...")

query_engine = hr_index.as_query_engine(
    similarity_top_k=3,
    response_mode="compact"
)

print("✅ Nexa RAG query engine ready")


# ============================================================
# 12. CHAT FUNCTION
# ============================================================

def ask_nexa(question):

    if not question or not question.strip():
        return "Please enter an HR question."

    question = question.strip()

    try:

        full_prompt = f"""
{SYSTEM_PROMPT}

User question:

{question}

Answer the user using only the retrieved DDS HR policy information.
"""

        response = query_engine.query(full_prompt)

        answer = str(response).strip()

        # ----------------------------------------------------
        # SOURCE EXTRACTION
        # ----------------------------------------------------

        sources = []

        try:
            for node in response.source_nodes:

                metadata = node.node.metadata or {}

                filename = (
                    metadata.get("file_name")
                    or metadata.get("filename")
                    or metadata.get("file_path")
                    or "DDS HR Policy"
                )

                page = (
                    metadata.get("page_label")
                    or metadata.get("page_number")
                    or metadata.get("page")
                )

                if page:
                    source = f"{filename} — Page {page}"
                else:
                    source = filename

                if source not in sources:
                    sources.append(source)

        except Exception:
            pass

        # ----------------------------------------------------
        # ADD SOURCES
        # ----------------------------------------------------

        if sources:

            source_text = "\n\n### Sources\n"

            for source in sources:
                source_text += f"- {source}\n"

            answer += source_text

        return answer

    except Exception as e:

        print("\n❌ ERROR WHILE ANSWERING:")
        print(e)

        return (
            "Sorry, Nexa could not process your question right now.\n\n"
            f"Technical error: {str(e)}"
        )


# ============================================================
# 13. FRONTEND
# ============================================================

print("\nBuilding Nexa frontend...")


with gr.Blocks(
    title="Nexa — DDS Enterprise HR Assistant"
) as demo:

    gr.Markdown(
        """
# 🤖 Nexa

### DDS Enterprise HR Assistant

Ask questions about DDS HR policies, procedures, and workplace guidelines.

**Nexa answers using the approved DDS HR knowledge base.**
"""
    )

    gr.Markdown(
        """
### 💬 Ask Nexa

Type your HR question below.
"""
    )

    question = gr.Textbox(
        label="HR Question",
        placeholder="Example: How many annual leave days do full-time employees receive?",
        lines=3
    )

    with gr.Row():

        ask_button = gr.Button(
            "Ask Nexa",
            variant="primary"
        )

        clear_button = gr.ClearButton(
            components=[question]
        )

    answer = gr.Markdown(
        label="Nexa's Answer"
    )

    ask_button.click(
        fn=ask_nexa,
        inputs=question,
        outputs=answer
    )

    question.submit(
        fn=ask_nexa,
        inputs=question,
        outputs=answer
    )


# ============================================================
# 14. START SERVER
# ============================================================

print("\n" + "=" * 60)
print("🔥 STARTING NEXA FRONTEND")
print("=" * 60)

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.getenv("PORT", 10000)),
    share=False
)

