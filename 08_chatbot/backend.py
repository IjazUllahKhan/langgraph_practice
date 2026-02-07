from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
import traceback

load_dotenv()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",  
    temperature=0.7
)

# Typed state for LangGraph
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Node function used by the graph
def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Set up sqlite connection & checkpointer
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Build graph and compile chatbot
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    """
    Returns a list of dicts: [{'thread_id': '<uuid-str>', 'thread_name': '...'}, ...]
    If there's no saved name for a thread, thread_name will be None.
    """
    threads = {}
    try:
        for checkpoint in checkpointer.list(None):
            cfg = checkpoint.config or {}
            # guard access - configurable might be missing
            configurable = cfg.get('configurable', {}) if isinstance(cfg, dict) else {}
            tid = configurable.get('thread_id')
            if not tid:
                continue
            meta = cfg.get('metadata', {}) if isinstance(cfg, dict) else {}
            name = meta.get('thread_name')
            # later checkpoints override earlier ones
            threads[str(tid)] = name
    except Exception as e:
        # if listing fails for some reason, print to help debugging
        print("retrieve_all_threads error:", e)
        traceback.print_exc()

    return [{"thread_id": tid, "thread_name": threads[tid]} for tid in threads]


def persist_thread_name(thread_id: str, name: str):
    """
    Persist `thread_name` into the checkpointer by writing a small state checkpoint
    with the same configurable.thread_id and metadata.thread_name. This uses chatbot.get_state
    to force writing a checkpoint entry containing the metadata.
    """
    try:
        chatbot.get_state(config={
            "configurable": {"thread_id": thread_id},
            "metadata": {"thread_name": name}
        })
    except Exception as e:
        print("persist_thread_name failed:", e)
        traceback.print_exc()
