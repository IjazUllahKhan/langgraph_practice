# streamlit_app.py
import streamlit as st
from backend import chatbot, retrieve_all_threads, persist_thread_name, get_thread_name
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# =========================== Utilities ===========================
def generate_thread_id():
    # return string to make mapping keys consistent
    return str(uuid.uuid4())

def name_from_first_message(text: str, max_len: int = 40):
    text = text.strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
    # ensure thread_names dict always has a key (value may be None)
    if thread_id not in st.session_state["thread_names"]:
        st.session_state["thread_names"][thread_id] = None

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    # Check if messages key exists in state values, return empty list if not
    try:
        values = getattr(state, "values", None)
        if isinstance(values, dict):
            return values.get("messages", []) or []
        if isinstance(state, dict):
            return state.get("messages", []) or []
    except Exception:
        pass
    return []

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    # retrieve_all_threads returns a list of thread ids (as in your original)
    st.session_state["chat_threads"] = retrieve_all_threads()

if "thread_names" not in st.session_state:
    # mapping thread_id -> name (None if not set)
    st.session_state["thread_names"] = {}

# populate thread_names from backend metadata for persisted threads
for tid in st.session_state["chat_threads"]:
    if tid not in st.session_state["thread_names"]:
        try:
            name = get_thread_name(tid)
            st.session_state["thread_names"][tid] = name
        except Exception:
            st.session_state["thread_names"][tid] = None

# ensure current thread is present in lists
add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")
# reverse so most recent appear first
for thread_id in st.session_state["chat_threads"][::-1]:
    display_name = st.session_state["thread_names"].get(thread_id)
    if not display_name:
        display_name = f"Chat {str(thread_id)[:8]}"
    # unique key per button
    if st.sidebar.button(display_name, key=f"btn_{thread_id}"):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages
        st.rerun()

# ============================ Main UI ============================
st.header("Conversation")

current_tid = st.session_state["thread_id"]
current_display = st.session_state["thread_names"].get(current_tid) or f"Chat {str(current_tid)[:8]}"
st.subheader(current_display)

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    current_tid = st.session_state["thread_id"]

    # Auto-name conversation from FIRST user message only (if not already named)
    if not st.session_state["thread_names"].get(current_tid):
        chat_name = name_from_first_message(user_input)
        st.session_state["thread_names"][current_tid] = chat_name
        # persist name in backend so it shows after restart
        try:
            persist_thread_name(current_tid, chat_name)
        except Exception as e:
            # log to console, but continue
            print("persist_thread_name error:", e)

    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"], "thread_name": st.session_state["thread_names"].get(current_tid)},
        "run_name": "chat_turn",
    }

    # Assistant streaming block (preserve tool-aware streaming)
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(f"🔧 Using `{tool_name}` …", expanded=True)
                    else:
                        status_holder["box"].update(label=f"🔧 Using `{tool_name}` …", state="running", expanded=True)

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(label="✅ Tool finished", state="complete", expanded=False)

    # Save assistant message
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
    # Rerun so sidebar updates and persisted names are visible
    st.rerun()
