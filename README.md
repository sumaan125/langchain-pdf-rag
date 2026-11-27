# Advanced RAG – Chat with your PDFs (LangChain)

Upload one or more PDFs, ask questions, and get answers with page sources.
Chats are saved, so you can come back and continue them later.

The pipeline is built almost entirely from **LangChain** components, while keeping
the advanced parts: structure-aware chunking, hybrid search, LLM reranking,
history-aware question rewriting, and proper math rendering.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://langchain-pdf-rag-inyfl9gftucdkwcztx6czt.streamlit.app/)

## Live demo

The app is deployed on Streamlit Community Cloud:
**https://langchain-pdf-rag-inyfl9gftucdkwcztx6czt.streamlit.app/**

If the app asks for a password, enter the one set by the owner. If it shows a "wake up" screen, click the button and wait a moment: free apps go to sleep when unused.

## Run the app

1. Clone the repository and open the folder:

```powershell
git clone https://github.com/YOUR_USERNAME/langchain-pdf-rag.git
cd langchain-pdf-rag
```

2. (Recommended) create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

3. Install the dependencies:

```powershell
pip install -r requirements.txt
```

4. Create a `.env` file next to `app.py` and add your keys:

```
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_ai_studio_key_here
```

5. Start the app:

```powershell
streamlit run app.py
```

Then open the link Streamlit prints (usually http://localhost:8501), upload one or more PDFs, click **Start chatting**, and ask questions.

## The code

| File | What it does |
|---|---|
| `config.py` | Models and numbers (chunk size, top-k). Change a model here. |
| `rag.py` | The whole RAG pipeline, written top to bottom with LangChain. |
| `history.py` | Save / load / delete chats as JSON files, one private folder per browser session in `data/chats/<session>/`. |
| `app.py` | The Streamlit screen. |

## Which LangChain feature does what

| Step | LangChain component |
|---|---|
| PDF loading | `PyMuPDFLoader` (`langchain_community`) |
| Chunking | `RecursiveCharacterTextSplitter` (paragraph → line → sentence → word) |
| Embeddings | `GoogleGenerativeAIEmbeddings` (`gemini-embedding-001`) |
| Vector store (saved to disk) | `FAISS` (`save_local` / `load_local`) |
| Keyword search | `BM25Retriever` |
| Hybrid search | `EnsembleRetriever` (Reciprocal Rank Fusion of vector + BM25) |
| Image analysis | `ChatGoogleGenerativeAI` (vision) via `.batch()`; images found with PyMuPDF |
| Reranking | `ContextualCompressionRetriever` + `LLMListwiseRerank` |
| Follow-up question rewriting | `create_history_aware_retriever` |
| Prompt + answer | `ChatPromptTemplate`, `create_stuff_documents_chain`, `ChatGroq` (streamed) |
| Chat memory | `HumanMessage` / `AIMessage` via `MessagesPlaceholder` |

## How an answer is made

1. **Chunking** – pages are split with `RecursiveCharacterTextSplitter` (~900 chars, 150 overlap), keeping paragraphs and sentences together; each chunk remembers its PDF and page.
2. **Embedding** – chunks go into a FAISS index saved in `data/indexes/`, so the same PDF is never embedded twice.
3. **Question rewriting** – a follow-up like *"how much did it cost?"* is rewritten into a standalone question using the chat history.
4. **Hybrid search** – FAISS (meaning) + BM25 (exact words, symbols) are merged with Reciprocal Rank Fusion.
5. **Reranking** – an LLM listwise reranker keeps the 4 best of the candidates.
6. **Answer** – `gpt-oss-120b` answers only from those chunks, with the last 6 messages as memory. If the PDF has no answer it says so.

## Images in PDFs

Pictures inside a PDF (charts, graphs, diagrams, tables saved as images, scanned figures) are
analysed during indexing: each one is sent to a vision model (`VISION_MODEL` in `config.py`,
Gemini) which describes it, transcribes any text and writes equations in LaTeX. That
description is stored as a chunk with the image's page number, so it is searched, reranked and
cited like normal text (shown as `[Image on this page] ...` in Sources).

Settings in `config.py`: `DESCRIBE_IMAGES` (turn off to skip), `MIN_IMAGE_SIZE` (ignores icons),
`MAX_IMAGES_PER_PDF` (cost guard). Repeated images (e.g. a logo on every page) are analysed once.
Only embedded raster images are analysed; charts drawn as vector graphics are not.
Analysis happens once per PDF set, then the saved index is reused. Re-upload PDFs that were
indexed before this feature to include their images.

## Math fixes

- **Extraction**: PDFs are read with PyMuPDF instead of `pypdf`, which keeps Unicode math symbols (∑ ≤ α √ …) intact. Line breaks are repaired without flattening paragraphs, so equations are no longer glued into one run of text.
- **Prompt**: the model rebuilds garbled equations from context and writes them as LaTeX (`$...$`, `$$...$$`).
- **Rendering**: `fix_math` converts `\( \)`, `\[ \]`, `\begin{equation}`/`align` and bare `[ latex ]` lines to the `$` form Streamlit renders. Code blocks and existing `$$` blocks are left alone, and it is safe to run twice.

## Privacy and password

- **Private chats:** every browser session gets a random id, and its chats are stored and listed only under that id. Visitors never see each other's chats. A page refresh starts a new session, so the previous session's chats are no longer shown.
- **Password (optional):** add `APP_PASSWORD=your_password` to `.env` (or to Streamlit Cloud *Secrets* as `APP_PASSWORD = "your_password"`) and the app asks for it before showing anything. If it is not set, there is no password prompt, which is convenient for local use.
## Deployment (Streamlit Community Cloud)

1. Push the project to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io) click **Create app**, choose the repository, branch `main` and main file `app.py`.
3. Under **Advanced settings**, pick Python **3.12** and paste your keys into **Secrets** (TOML format):

```toml
GROQ_API_KEY = "your_groq_key"
GOOGLE_API_KEY = "your_google_ai_studio_key"
APP_PASSWORD = "your_password"
```

4. Click **Deploy**. Later `git push` updates redeploy the app automatically.

Note: the server's disk is temporary, so saved chats and indexes are lost when the app restarts.

## Notes

- Works on text PDFs. Scanned (image-only) PDFs need OCR first.
- Equations that a PDF stores as images or exotic fonts can still be extracted imperfectly; the model reconstructs them as best it can from the surrounding text.
- Indexes saved by the previous version (`.json` + `.npy`) are not reused; upload the PDF again to rebuild.
