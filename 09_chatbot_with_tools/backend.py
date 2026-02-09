from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
import os

load_dotenv()

# ======================
# 1. LLM
# ======================
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
)

# ======================
# 2. Tools
# ======================
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool(description="Perform basic arithmetic operations: add, sub, mul, div")
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    if operation == "add":
        result = first_num + second_num
    elif operation == "sub":
        result = first_num - second_num
    elif operation == "mul":
        result = first_num * second_num
    elif operation == "div":
        if second_num == 0:
            return {"error": "Division by zero"}
        result = first_num / second_num
    else:
        return {"error": "Unsupported operation"}

    return {"result": result}

@tool(description="Get the latest stock price for a given stock symbol")
def get_stock_price(symbol: str) -> dict:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    return r.json()

tools = [search_tool, calculator, get_stock_price]
llm_with_tools = llm.bind_tools(tools)

# ======================
# 3. State
# ======================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ======================
# 4. Nodes
# ======================
def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

# ======================
# 5. SQLite
# ======================
DB_PATH = os.path.join(os.getcwd(), "chatbot.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)

# Thread name table
conn.execute("""
CREATE TABLE IF NOT EXISTS thread_names (
    thread_id TEXT PRIMARY KEY,
    name TEXT
)
""")
conn.commit()

def persist_thread_name(thread_id: str, name: str):
    conn.execute(
        "INSERT OR REPLACE INTO thread_names (thread_id, name) VALUES (?, ?)",
        (str(thread_id), name),
    )
    conn.commit()

def get_thread_name(thread_id: str):
    cur = conn.execute(
        "SELECT name FROM thread_names WHERE thread_id = ?",
        (str(thread_id),),
    )
    row = cur.fetchone()
    return row[0] if row else None

def retrieve_all_threads():
    threads = set()
    for checkpoint in checkpointer.list(None):
        threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(threads)

# ======================
# 6. Graph
# ======================
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)
