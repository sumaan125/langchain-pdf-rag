"""All settings in one place. Change a model here and the whole app follows."""
import os
from dotenv import load_dotenv

load_dotenv()  # reads GROQ_API_KEY and GOOGLE_API_KEY from the .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
APP_PASSWORD = os.getenv("APP_PASSWORD")  # optional: if set, visitors must enter it

CHAT_MODEL = "openai/gpt-oss-120b"    # Groq: writes the final answer
FAST_MODEL = "openai/gpt-oss-20b"     # Groq: cheap helper (rewrite question, rerank)
EMBED_MODEL = "gemini-embedding-001"  # Google: turns text into vectors

VISION_MODEL = "gemini-3.5-flash"     # Google: describes images found inside PDFs
DESCRIBE_IMAGES = True   # set False to skip image analysis (faster, no extra API calls)
MIN_IMAGE_SIZE = 100     # ignore images smaller than this many pixels (icons, bullets)
MAX_IMAGES_PER_PDF = 30  # cost guard: analyse at most this many images per PDF

CHUNK_SIZE = 900     # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between neighbouring chunks
CANDIDATES = 12       # chunks each retriever (vector, BM25) returns, before reranking
TOP_K = 4             # chunks finally given to the LLM
HISTORY_TURNS = 6     # how many past messages the LLM sees

DATA_DIR = "data"     # chats and indexes are saved here
