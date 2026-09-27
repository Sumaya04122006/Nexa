# ============================================================
# NEXA — DDS ENTERPRISE HR ASSISTANT
# Gemini + LlamaIndex + Pinecone + Gradio
# Professional SaaS-style frontend
# ============================================================

import os
import base64
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

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
GEMINI_MODEL = "gemini-3.1-flash-lite"

EMBEDDING_MODEL = "gemini-embedding-001"

PINECONE_INDEX_NAME = "nexa-gemini-hr"
PINECONE_NAMESPACE = "nexa-hr-v1"

PDF_FOLDER = Path("DDS_HR_Handbook")
ASSETS_FOLDER = Path(__file__).parent / "assets"
LOGO_FILE = ASSETS_FOLDER / "dds_logo.png"


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
        "├── assets/\n"
        "│   └── dds_logo.png\n"
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

def ask_nexa(question, history):
    """
    history: list of dicts, Gradio 'messages' format
    Returns updated history + clears the textbox.
    """

    history = history or []

    if not question or not question.strip():
        return history, ""

    question = question.strip()
    history.append({"role": "user", "content": question})

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

            source_text = "\n\n---\n**📄 Sources**\n"

            for source in sources:
                source_text += f"- {source}\n"

            answer += source_text

        history.append({"role": "assistant", "content": answer})

    except Exception as e:

        print("\n❌ ERROR WHILE ANSWERING:")
        print(e)

        error_msg = (
            "Sorry, Nexa could not process your question right now.\n\n"
            f"**Technical error:** {str(e)}"
        )
        history.append({"role": "assistant", "content": error_msg})

    return history, ""


def clear_chat():
    return [], ""


# ============================================================
# 13. FRONTEND — PROFESSIONAL SAAS-STYLE UI
# ============================================================

print("\nBuilding Nexa frontend...")

# Encode the Decoding Data Science logo as a data URI so the header
# is fully self-contained (no broken image links if the file moves).
if LOGO_FILE.exists():
    _logo_bytes = LOGO_FILE.read_bytes()
    LOGO_B64 = base64.b64encode(_logo_bytes).decode("utf-8")
    LOGO_SRC = f"data:image/png;base64,{LOGO_B64}"
else:
    LOGO_SRC = ""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
    --nexa-bg: #f5f7fb;
    --nexa-panel: #ffffff;
    --nexa-border: #e5e9f0;
    --nexa-navy: #0f172a;
    --nexa-navy-light: #1e293b;
    --nexa-accent: #2f5fff;
    --nexa-accent-dark: #1d3fd1;
    --nexa-text: #101828;
    --nexa-muted: #667085;
    --nexa-success: #16a34a;
}

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

.gradio-container{
    background: var(--nexa-bg) !important;
    max-width: 1180px !important;
    margin: 0 auto !important;
}

footer{ display:none !important; }

/* ---------- Header bar ---------- */
#nexa-header{
    background: linear-gradient(135deg, var(--nexa-navy) 0%, var(--nexa-navy-light) 100%);
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
}
#nexa-header .nexa-brand{
    display: flex;
    align-items: center;
    gap: 14px;
}
#nexa-header .nexa-mark{
    width: 46px;
    height: 46px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--nexa-accent), #7c3aed);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    box-shadow: 0 4px 12px rgba(47, 95, 255, 0.4);
    flex-shrink: 0;
}
#nexa-header .nexa-title{
    color: #ffffff;
    font-size: 20px;
    font-weight: 800;
    line-height: 1.2;
    margin: 0;
}
#nexa-header .nexa-subtitle{
    color: #a8b3cf;
    font-size: 12.5px;
    font-weight: 500;
    margin: 2px 0 0 0;
    display: flex;
    align-items: center;
    gap: 6px;
}
#nexa-header .nexa-dot{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 0 3px rgba(34,197,94,0.25);
    display: inline-block;
}
#nexa-header .nexa-partner-logo{
    height: 46px;
    width: auto;
    border-radius: 8px;
    background: #ffffff;
    padding: 4px 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}

/* ---------- Layout panels ---------- */
#nexa-sidebar{
    background: var(--nexa-panel);
    border: 1px solid var(--nexa-border);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(16,24,40,0.04);
}
#nexa-sidebar h3{
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--nexa-muted);
    font-weight: 700;
    margin: 0 0 12px 0;
}
#nexa-sidebar .nexa-chip{
    display:block;
    width:100%;
    text-align:left;
    background: #f5f7fb;
    border: 1px solid var(--nexa-border);
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-size: 13px;
    color: var(--nexa-text);
    cursor: pointer;
}

#nexa-chat-panel{
    background: var(--nexa-panel);
    border: 1px solid var(--nexa-border);
    border-radius: 16px;
    box-shadow: 0 1px 3px rgba(16,24,40,0.04);
    padding: 6px 6px 14px 6px;
}

#nexa-chatbot{
    border: none !important;
    background: transparent !important;
}
#nexa-chatbot .message.user{
    background: var(--nexa-accent) !important;
    color: #fff !important;
    border-radius: 14px 14px 2px 14px !important;
}
#nexa-chatbot .message.bot{
    background: #f5f7fb !important;
    color: var(--nexa-text) !important;
    border-radius: 14px 14px 14px 2px !important;
    border: 1px solid var(--nexa-border) !important;
}

#nexa-input-row{
    padding: 10px 14px 0 14px;
}
#nexa-input-row textarea{
    border-radius: 12px !important;
    border: 1.5px solid var(--nexa-border) !important;
    background: #fbfcfe !important;
}
#nexa-input-row textarea:focus{
    border-color: var(--nexa-accent) !important;
}

#nexa-send-btn{
    background: var(--nexa-accent) !important;
    color: #fff !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    border: none !important;
}
#nexa-send-btn:hover{ background: var(--nexa-accent-dark) !important; }

#nexa-clear-btn{
    border-radius: 12px !important;
    font-weight: 600 !important;
    border: 1.5px solid var(--nexa-border) !important;
    background: #fff !important;
    color: var(--nexa-muted) !important;
}

#nexa-footer{
    text-align: center;
    color: var(--nexa-muted);
    font-size: 12px;
    margin-top: 16px;
    padding-bottom: 6px;
}
"""

EXAMPLE_QUESTIONS = [
    "How many annual leave days do full-time employees receive?",
    "What is the process for requesting sick leave?",
    "What is DDS's remote work policy?",
    "How do I report a workplace grievance?",
]

with gr.Blocks(
    title="Nexa — DDS Enterprise HR Assistant",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
) as demo:

    # ---------------- Header ----------------
    gr.HTML(f"""
    <div id="nexa-header">
        <div class="nexa-brand">
            <div class="nexa-mark">🤖</div>
            <div>
                <p class="nexa-title">Nexa · HR Assistant</p>
                <p class="nexa-subtitle"><span class="nexa-dot"></span>DDS Enterprise · Knowledge base connected</p>
            </div>
        </div>
        {f'<img class="nexa-partner-logo" src="{LOGO_SRC}" alt="Decoding Data Science" />' if LOGO_SRC else ''}
    </div>
    """)

    with gr.Row(equal_height=False):

        # ---------------- Sidebar ----------------
        with gr.Column(scale=1, min_width=260, elem_id="nexa-sidebar"):
            gr.Markdown("### About Nexa")
            gr.Markdown(
                "Nexa answers questions using DDS's approved HR policy "
                "documents only. It will not provide confidential employee "
                "data or salary information."
            )
            gr.HTML("<h3 style='margin-top:22px;'>Try asking</h3>")
            example_buttons = [gr.Button(q, elem_classes="nexa-chip", size="sm")
                                for q in EXAMPLE_QUESTIONS]

            gr.HTML(
                "<h3 style='margin-top:22px;'>Scope</h3>"
                "<p style='font-size:13px;color:var(--nexa-muted);line-height:1.5;'>"
                "Leave &amp; time off · Benefits · Conduct policy · Onboarding · "
                "Workplace procedures</p>"
            )

        # ---------------- Chat panel ----------------
        with gr.Column(scale=3, elem_id="nexa-chat-panel"):

            chatbot = gr.Chatbot(
                elem_id="nexa-chatbot",
                type="messages",
                height=460,
                show_label=False,
                avatar_images=(None, None),
                placeholder="👋 Hi, I'm Nexa. Ask me anything about DDS HR policy.",
            )

            with gr.Row(elem_id="nexa-input-row"):
                question = gr.Textbox(
                    show_label=False,
                    placeholder="Type your HR question here…",
                    lines=1,
                    scale=8,
                    container=False,
                )
                send_btn = gr.Button("Send", elem_id="nexa-send-btn", scale=1)

            with gr.Row(elem_id="nexa-input-row"):
                clear_btn = gr.Button("Clear conversation", elem_id="nexa-clear-btn", size="sm")

    gr.HTML(
        f"<div id='nexa-footer'>Nexa can make mistakes. Verify important "
        f"information with HR. · © {datetime.now().year} DDS Enterprise</div>"
    )

    # ---------------- Wiring ----------------
    send_btn.click(fn=ask_nexa, inputs=[question, chatbot], outputs=[chatbot, question])
    question.submit(fn=ask_nexa, inputs=[question, chatbot], outputs=[chatbot, question])
    clear_btn.click(fn=clear_chat, inputs=None, outputs=[chatbot, question])

    for btn in example_buttons:
        btn.click(fn=ask_nexa, inputs=[btn, chatbot], outputs=[chatbot, question])


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
