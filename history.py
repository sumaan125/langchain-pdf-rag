"""Saving and loading chats. One chat = one small JSON file.
Every browser session has its own folder (data/chats/<owner>/), so visitors never see
each other's chats."""
import json
import os
import time
import uuid

from config import DATA_DIR

CHAT_ROOT = os.path.join(DATA_DIR, "chats")
os.makedirs(CHAT_ROOT, exist_ok=True)


def _dir(owner):
    path = os.path.join(CHAT_ROOT, owner)
    os.makedirs(path, exist_ok=True)
    return path


def _path(owner, chat_id):
    return os.path.join(_dir(owner), f"{chat_id}.json")


def new_owner():
    """A random id for one browser session. Nobody can guess another session's folder."""
    return uuid.uuid4().hex


def new_chat():
    return {
        "id": uuid.uuid4().hex[:8],
        "title": "New chat",
        "created": time.time(),
        "files": [],       # names of the PDFs used in this chat
        "index_id": None,  # which saved index belongs to this chat
        "messages": [],    # {"role": "user"/"assistant", "content": ..., "sources": [...]}
    }


def save_chat(owner, chat):
    with open(_path(owner, chat["id"]), "w", encoding="utf-8") as f:
        json.dump(chat, f, indent=2, ensure_ascii=False)


def load_chat(owner, chat_id):
    with open(_path(owner, chat_id), encoding="utf-8") as f:
        return json.load(f)


def list_chats(owner):
    """This session's saved chats, newest first."""
    folder = _dir(owner)
    chats = [load_chat(owner, n[:-5]) for n in os.listdir(folder) if n.endswith(".json")]
    return sorted(chats, key=lambda c: c["created"], reverse=True)


def delete_chat(owner, chat_id):
    path = _path(owner, chat_id)
    if os.path.exists(path):
        os.remove(path)
