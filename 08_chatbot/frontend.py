# streamlit_app.py
import streamlit as st
from backend import chatbot, retrieve_all_threads, persist_thread_name
from langchain_core.messages import HumanMessage
import uuid

# ------------------- Helper / utility functions -------------------

def generate_thread_id():
    return str(uuid.uuid4())

def name_from_first_message(text: str, max_len: int = 40):
    text = text.strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."

def load_conversation(thread_id: str):
    """
    Returns a list of BaseMessage objects (or empty list).
    """
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Some LangGraph states may expose values as `.values` mapping
    # We try to be defensive here.
    try:
        values = getattr(state, "values", None)
        if isinstance(values, dict):
            return values.get('messages', []) or []
        # fallback: maybe state itself is a dict-like
        if isinstance(state, dict):
            return state.get('messages', []) or []
    except Exception:
        pass
    return []

# ------------------- Streamlit session init -----------------------

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'thread_names' not in st.session_state:
    st.session_state['thread_names'] = {}

# On first load, populate persisted threads from backend
if not st.session_state['chat_threads']:
    backend_threads = retrieve_all_threads()
    for t in backend_threads:
        tid = str(t['thread_id'])
        if tid not in st.session_state['chat_threads']:
            st.session_state['chat_threads'].append(tid)
        if t.get('thread_name'):
            st.session_state['thread_names'][tid] = t['thread_name']

# Make sure current thread exists in lists
current_tid = st.session_state['thread_id']
if current_tid not in st.session_state['chat_threads']:
    st.session_state['chat_threads'].append(current_tid)

# ------------------- Sidebar UI -----------------------------------

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    # create a fresh thread id and reset in-session history only
    new_tid = generate_thread_id()
    st.session_state['thread_id'] = new_tid
    st.session_state['message_history'] = []
    # don't assign a name yet — will be set on first user message
    if new_tid not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(new_tid)
    st.rerun()

st.sidebar.header('My Conversations')

# Reverse order so most recent appear on top
for tid in st.session_state['chat_threads'][::-1]:
    display_name = st.session_state['thread_names'].get(tid, f"Chat {tid[:8]}")
    btn_key = f"select_{tid}"
    if st.sidebar.button(display_name, key=btn_key):
        # load messages from backend and set message_history for the UI
        st.session_state['thread_id'] = tid
        messages = load_conversation(tid)
        temp_messages = []
        for msg in messages:
            # msg is a BaseMessage-like object; we compare types defensively
            try:
                from langchain_core.messages import HumanMessage as LC_HumanMessage
                if isinstance(msg, LC_HumanMessage):
                    role = 'user'
                else:
                    role = 'assistant'
            except Exception:
                # fallback: check attribute 'type' or 'role' if present
                role = 'user' if getattr(msg, "type", "") == "human" else 'assistant'
            # msg.content should exist
            content = getattr(msg, "content", str(msg))
            temp_messages.append({'role': role, 'content': content})
        st.session_state['message_history'] = temp_messages
        st.rerun()

# ------------------- Main UI --------------------------------------

st.header("Conversation")

current_tid = st.session_state['thread_id']
current_name = st.session_state['thread_names'].get(current_tid, f"Chat {current_tid[:8]}")
st.subheader(current_name)

# Display conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# User input area
user_input = st.chat_input('Type here')

if user_input:
    current_tid = st.session_state['thread_id']

    # Auto-name conversation from FIRST user message only (if not already named)
    if current_tid not in st.session_state['thread_names']:
        chat_name = name_from_first_message(user_input)
        st.session_state['thread_names'][current_tid] = chat_name
        # Persist to backend checkpointer so name survives restarts
        persist_thread_name(current_tid, chat_name)

    # Append user message to history and show it
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # Prepare config including metadata so checkpoint saved with name
    CONFIG = {
        "configurable": {"thread_id": current_tid},
        "metadata": {
            "thread_id": current_tid,
            "thread_name": st.session_state['thread_names'].get(current_tid)
        },
        "run_name": "chat_turn",
    }

    # Stream assistant reply
    with st.chat_message('assistant'):
        try:
            ai_message = st.write_stream(
                message_chunk.content
                for message_chunk, metadata in chatbot.stream(
                    {'messages': [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode='messages'
                )
            )
        except Exception as e:
            # fallback: call non-streaming if stream fails
            st.text("Assistant error: " + str(e))
            ai_message = "Assistant failed to produce a response."

    # Append assistant message to session history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    # After a new turn, rerun so sidebar names reflect changes
    st.rerun()
