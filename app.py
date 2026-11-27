"""Chat with your PDFs. Run:  streamlit run app.py"""
import hmac

import streamlit as st

import config
import history

st.set_page_config(page_title="PDF Chat", page_icon="📄")


# ---------------------------------------------------------------- password gate
def check_password():
    """If APP_PASSWORD is set, nothing else is shown (or imported) until it is entered."""
    if not config.APP_PASSWORD or st.session_state.get("authed"):
        return
    st.title("🔒 PDF Chat")
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered.encode(), config.APP_PASSWORD.encode()):
            st.session_state.authed = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()


check_password()
import rag  # noqa: E402  (imported after the gate so no API clients start for strangers)

# st.session_state keeps values while the page re-runs after every click.
# `owner` is a random id for this browser session: it keeps chats private.
if "owner" not in st.session_state:
    st.session_state.owner = history.new_owner()
owner = st.session_state.owner

if "chat" not in st.session_state:
    st.session_state.chat = history.new_chat()
    st.session_state.index = None
chat = st.session_state.chat


def reset_chat():
    st.session_state.chat = history.new_chat()
    st.session_state.index = None


def open_chat(chat_id):
    """Load an old chat together with its saved index."""
    st.session_state.chat = history.load_chat(owner, chat_id)
    saved = st.session_state.chat["index_id"]
    st.session_state.index = rag.load_index(saved) if saved else None


def show_sources(sources):
    with st.expander("Sources"):
        for s in sources:
            st.caption(f"{s['file']} — page {s['page']}")
            st.text(s["text"])  # plain text, so a "$" in the PDF is not read as math


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📄 PDF Chat")
    if st.button("➕ New chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.subheader("Past chats")
    for old in history.list_chats(owner):
        if not old["messages"]:
            continue  # skip chats where nothing was asked yet
        col1, col2 = st.columns([5, 1])
        if col1.button(old["title"][:28], key=f"open_{old['id']}", use_container_width=True):
            open_chat(old["id"])
            st.rerun()
        if col2.button("🗑", key=f"del_{old['id']}"):
            history.delete_chat(owner, old["id"])
            if old["id"] == chat["id"]:
                reset_chat()
            st.rerun()

# ---------------------------------------------------------------- upload step
st.title("Chat with your PDFs")

if st.session_state.index is None:
    if chat["messages"]:
        st.warning("The saved index for this chat is missing. Upload the PDF(s) again to continue.")
    uploads = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)
    if uploads and st.button("Start chatting", type="primary"):
        with st.spinner("Reading and indexing... (slow only the first time)"):
            index_id, index = rag.build_index({f.name: f for f in uploads})
        chat["files"] = [f.name for f in uploads]
        chat["index_id"] = index_id
        st.session_state.index = index
        history.save_chat(owner, chat)
        st.rerun()
else:
    st.caption("Files: " + ", ".join(chat["files"]))

# ---------------------------------------------------------------- chat
for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(rag.fix_math(msg["content"]))
        if msg.get("sources"):
            show_sources(msg["sources"])

question = st.chat_input("Ask something about your PDF", disabled=st.session_state.index is None)
if question:
    past = list(chat["messages"])  # the history BEFORE this question
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            docs = rag.retrieve(st.session_state.index, question, past)
            sources = rag.source_dicts(docs)
        box, reply = st.empty(), ""
        for piece in rag.answer(question, past, docs):  # type the answer live
            reply += piece
            box.markdown(rag.fix_math(reply))
        reply = rag.fix_math(reply)
        show_sources(sources)

    chat["messages"].append({"role": "user", "content": question})
    chat["messages"].append({"role": "assistant", "content": reply, "sources": sources})
    if chat["title"] == "New chat":
        chat["title"] = question[:40]  # the first question names the chat
    history.save_chat(owner, chat)
    st.rerun()
