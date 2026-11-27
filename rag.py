"""The RAG pipeline, built from LangChain components. Read it top to bottom:

   PDF -> pages -> chunks -> FAISS vectors (saved)          [build_index]
   question -> rewrite -> hybrid search (FAISS + BM25) -> LLM rerank   [retriever]
   chunks + chat history -> streamed answer                  [answer]
"""
import base64
import hashlib
import os
import re
import tempfile
import warnings

import pymupdf

from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import LLMListwiseRerank
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

INDEX_DIR = os.path.join(config.DATA_DIR, "indexes")
os.makedirs(INDEX_DIR, exist_ok=True)

# ---------------------------------------------------------------- models
embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBED_MODEL, google_api_key=config.GOOGLE_API_KEY)
fast_llm = ChatGroq(model=config.FAST_MODEL, api_key=config.GROQ_API_KEY,
                    temperature=0.2, reasoning_effort="low")
chat_llm = ChatGroq(model=config.CHAT_MODEL, api_key=config.GROQ_API_KEY,
                    temperature=0.2, reasoning_effort="low", streaming=True)
# Vision model: reads the pictures inside PDFs (charts, diagrams, scanned figures).
vision_llm = ChatGoogleGenerativeAI(model=config.VISION_MODEL, google_api_key=config.GOOGLE_API_KEY,
                                    temperature=0, max_retries=6, timeout=120)


# ---------------------------------------------------------------- 1. read + chunk
# PyMuPDF keeps Unicode math symbols (∑ ≤ α ∫ ...) intact; pypdf often mangled them.
def read_pdf(path, name):
    """One Document per page, tagged with the PDF name and page number."""
    docs = PyMuPDFLoader(path).load()
    for d in docs:
        d.page_content = clean_pdf_text(d.page_content)
        d.metadata = {"file": name, "page": d.metadata.get("page", 0) + 1}
    return docs


IMAGE_PROMPT = (
    "This image comes from page {page} of a PDF document. Describe it so someone who "
    "cannot see it could answer questions about it. If it is a chart or graph: give the "
    "type, axes, units, series and key values/trends. If it is a table: transcribe it. "
    "If it is a diagram: explain the parts and how they connect. Transcribe any text in it, "
    "and write any equations in LaTeX ($...$). Be factual; do not guess."
)


_incomplete = []


def describe_images(path, name):
    """Find the pictures inside a PDF and turn each into a text Document with a vision LLM,
    so charts, diagrams and figures become searchable like normal text."""
    pdf = pymupdf.open(path)
    requests, metas, seen = [], [], set()
    for page in pdf:
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen:  # the same logo repeated on every page
                continue
            seen.add(xref)
            if img[2] < config.MIN_IMAGE_SIZE or img[3] < config.MIN_IMAGE_SIZE:
                continue  # icons, bullets, rules
            try:
                pix = pymupdf.Pixmap(pdf, xref)
                if pix.n - pix.alpha >= 4:  # CMYK -> RGB so PNG can store it
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                data = base64.b64encode(pix.tobytes("png")).decode()
            except Exception:
                continue
            requests.append([HumanMessage(content=[
                {"type": "text", "text": IMAGE_PROMPT.format(page=page.number + 1)},
                {"type": "image_url", "image_url": f"data:image/png;base64,{data}"},
            ])])
            metas.append({"file": name, "page": page.number + 1, "type": "image"})
            if len(requests) >= config.MAX_IMAGES_PER_PDF:
                break
        if len(requests) >= config.MAX_IMAGES_PER_PDF:
            break
    pdf.close()

    docs = []
    incomplete = _incomplete  # names of PDFs where an image failed (index is then not cached)
    # return_exceptions: one failed image must not break the whole upload
    results = vision_llm.batch(requests, config={"max_concurrency": 4}, return_exceptions=True)
    for meta, res in zip(metas, results):
        if isinstance(res, Exception):
            warnings.warn(f"Could not analyse an image on {name} p.{meta['page']}: {res}")
            incomplete.append(name)
            continue
        text = getattr(res, "content", "")
        if isinstance(text, list):
            text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in text)
        if isinstance(text, str) and text.strip():
            docs.append(Document(page_content="[Image on this page] " + text.strip(), metadata=meta))
    return docs


def clean_pdf_text(text):
    """Repair PDF line breaking without destroying equations or paragraphs."""
    text = text.replace("\u00ad", "")                       # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)            # re-join hyphenated words
    text = re.sub(r"(?<![.!?:;\n])\n(?!\n)", " ", text)     # single line breaks inside a sentence
    return re.sub(r"[ \t]+", " ", text).strip()


# Splits on paragraph, then line, then sentence, then word: the biggest natural unit that fits.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    add_start_index=True,
)


def tokenize(text):
    """Keep symbols as tokens too, so formulas like 'x^2' or '≤' are searchable."""
    return re.findall(r"\w+|[^\w\s]", text.lower())


# ---------------------------------------------------------------- 2. the index
def _index_path(index_id):
    return os.path.join(INDEX_DIR, index_id)


def load_index(index_id):
    """Open a saved index (a list of chunks + FAISS vectors). None if it does not exist."""
    path = _index_path(index_id)
    if not os.path.isdir(path):
        return None
    store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return Index(list(store.docstore._dict.values()), store)


class Index:
    """Chunks + a FAISS vector store. The BM25 index is rebuilt in memory (it is fast)."""

    def __init__(self, chunks, store):
        self.chunks = chunks
        self.store = store


def build_index(files):
    """files = {pdf_name: uploaded_file}. Embeds once, then reuses the saved copy."""
    digest = hashlib.md5()
    for name in sorted(files):
        digest.update(files[name].getvalue())
    index_id = digest.hexdigest()[:12]

    saved = load_index(index_id)
    if saved:
        return index_id, saved

    pages = []
    for name, f in files.items():
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(f.getvalue())
        try:
            pages += read_pdf(tmp.name, name)
            if config.DESCRIBE_IMAGES:
                pages += describe_images(tmp.name, name)
        finally:
            os.remove(tmp.name)

    chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]
    if not chunks:
        raise ValueError("No text found. Is the PDF a scanned image?")

    store = FAISS.from_documents(chunks, embeddings)
    if _incomplete:
        # Some images failed: use this index now, but do not cache it, so the next
        # upload retries the images instead of reusing an index that lacks them.
        _incomplete.clear()
    else:
        store.save_local(_index_path(index_id))
    return index_id, Index(chunks, store)


# ---------------------------------------------------------------- 3. retriever
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    ("human", "Rewrite the last question so it can be understood without the chat. "
              "Replace words like 'it', 'that', 'they' with what they refer to. "
              "Output ONLY the rewritten question."),
])


def build_retriever(index):
    """follow-up question -> standalone question -> hybrid search -> LLM rerank."""
    vector = index.store.as_retriever(search_kwargs={"k": config.CANDIDATES})
    keyword = BM25Retriever.from_documents(index.chunks, preprocess_func=tokenize)
    keyword.k = config.CANDIDATES

    # Ensemble = Reciprocal Rank Fusion of meaning-search and exact-word search.
    hybrid = EnsembleRetriever(retrievers=[vector, keyword], weights=[0.5, 0.5])

    reranker = LLMListwiseRerank.from_llm(fast_llm, top_n=config.TOP_K)
    reranked = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=hybrid)

    # Rewrites follow-ups using history (skips the LLM call when history is empty).
    return create_history_aware_retriever(fast_llm, reranked, CONDENSE_PROMPT)


def to_messages(history):
    recent = history[-config.HISTORY_TURNS:]
    return [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
            for m in recent]


def retrieve(index, question, history):
    """Returns the best chunks (Documents) for the question."""
    retriever = build_retriever(index)
    try:
        docs = retriever.invoke({"input": question, "chat_history": to_messages(history)})
    except Exception:
        # If the LLM reranker fails (bad structured output), fall back to plain hybrid order.
        base = EnsembleRetriever(
            retrievers=[index.store.as_retriever(search_kwargs={"k": config.CANDIDATES}),
                        BM25Retriever.from_documents(index.chunks, preprocess_func=tokenize)],
            weights=[0.5, 0.5])
        docs = base.invoke(question)[:config.TOP_K]
    return docs[:config.TOP_K]


def source_dicts(docs):
    """Plain dicts so chats can be saved as JSON."""
    return [{"file": d.metadata.get("file"), "page": d.metadata.get("page"),
             "text": d.page_content} for d in docs]


# ---------------------------------------------------------------- 4. answer
SYSTEM_PROMPT = (
    "You answer questions about PDF documents. Use ONLY the context below. "
    'If the answer is not in the context, say "I could not find that in the document." '
    "Never invent facts. Keep the answer clear and concise. "
    "Mention the page like (p. 3) when you use a fact. "
    "The context comes from a PDF, so equations may look garbled (Unicode symbols, broken "
    "fractions, split sub/superscripts). Rebuild them carefully from the context. "
    "Write ALL math in LaTeX: $...$ for inline and $$...$$ for display equations. "
    "Never put math in plain text or in code blocks. "
    "Never use \\( \\) or \\[ \\] delimiters.\n\nCONTEXT:\n{context}"
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

DOC_PROMPT = ChatPromptTemplate.from_template("[{file} p.{page}]\n{page_content}")

# create_stuff_documents_chain = "paste all chunks into the prompt, then call the LLM".
answer_chain = create_stuff_documents_chain(chat_llm, ANSWER_PROMPT, document_prompt=DOC_PROMPT)


def answer(question, history, docs):
    """Returns a stream of text pieces so the UI can type the answer live."""
    for piece in answer_chain.stream(
            {"input": question, "chat_history": to_messages(history), "context": docs}):
        if piece:
            yield piece


# ---------------------------------------------------------------- 5. math rendering
_ENV = r"(?:equation|align|aligned|gather|multline|eqnarray)\*?"


def fix_math(text):
    r"""Streamlit draws math only as $...$ or $$...$$. LLMs also write \(..\), \[..\],
    \begin{equation}..., or a bare line like [ \frac{a}{b} ]. Convert them all.
    Code blocks and existing $$..$$ blocks are left untouched, so it is safe to run twice."""
    protected = re.compile(r"(```.*?```|`[^`\n]+`|\$\$.+?\$\$)", re.S)
    parts = protected.split(text)
    for i in range(0, len(parts), 2):  # even parts = text outside protected blocks
        parts[i] = _fix_segment(parts[i])
    return "".join(parts)


def _display(m):
    return "\n\n$$" + m.group(1).strip() + "$$\n\n"


def _fix_segment(t):
    t = re.sub(r"\\\[(.+?)\\\]", _display, t, flags=re.S)
    t = re.sub(r"\\\((.+?)\\\)", lambda m: "$" + m.group(1).strip() + "$", t, flags=re.S)
    t = re.sub(r"(\\begin\{" + _ENV + r"\}.+?\\end\{" + _ENV + r"\})", _display, t, flags=re.S)
    # a line holding only "[ latex ]" -> display math
    t = re.sub(r"(?m)^[ \t]*\[\s*(.*\\[a-zA-Z]+.*?)\s*\][ \t]*$", _display, t)
    return t
