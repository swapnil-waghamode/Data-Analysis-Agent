from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

# Dataframe
df:pd.DataFrame = None 

def set_df(dataframe:pd.DataFrame):
  global df 

  df = dataframe

# ============== Defining Tools =================================

@tool
def get_data_info():
  """Get columns, data types, shape and first 5 rows of the uploaded csv file."""

  if df is None:
    return f"No CSV loaded."

  return f"""
   Shape : {df.shape},
   Columns & Types : {df.dtypes.to_string()},
   First 5 rows : {df.head().to_string()}
"""

@tool
def filter_tool(condition:str):
  """
  Filter rows using pandas query condition.
  Example: 'Age >30 and Department == 'Engineering''
  """

  if df is None:
    return f"No CSV loaded."

  try:
    result = df.query(condition)
    return f"Found {len(result)} rows: {result.to_string()}"
  except Exception as e:
    return f"Error: {e}"


@tool
def analyze_data(expression:str):
  """
  Run a pandas expression on the dataframe(referred to as 'df').
  Examples:
  df['salary'].mean()
  df.groupby('department')['salary'].mean()
  df['city'].value_counts()
  """

  if df is None:
    return f"No CSV loaded."

  try:
    result = eval(expression)
    return str(result)
  except Exception as e:
    return f"Error: {e}"

# ========================================================

TOOLS = [get_data_info, filter_tool, analyze_data]

MODEL = os.getenv("MODEL")
llm = ChatGroq(model=MODEL)

system_prompt = """
You are a data analyst assistant.You help users explore and analyze CSV data.
Use the tools to answer questions:
  - get_data_info -> see columns, types and sample rows.
  - filter_tool -> filter rows by condition.
  - analyze_data -> run pandas expression for stats and aggregations.
Always use get_data_info first if you are ansure about column names.
"""

MEMORY = InMemorySaver()

agent = create_agent(
  model=llm,
  system_prompt=system_prompt,
  tools=TOOLS,
  checkpointer=MEMORY
)

def run_agent(query:str):

  res = agent.invoke(
    {"messages":[{"role":"user","content":query}]},
    {"configurable":{"thread_id":"1"}}
  )

  ans =res["messages"][-1].content
  return ans

# ================ Building UI ==================

st.header(":green[📊Data Analysis Agent]")
st.subheader(":rainbow[Upload any csv file and get the analysis.]")

file = st.file_uploader(label="Select your CSV file", type="csv")

if file:
  df = pd.read_csv(file)
  set_df(df)

  st.dataframe(df.head())
  st.success(f"Loaded {df.shape[0]} rows.")


if "messages" not in st.session_state:
  st.session_state.messages = []

for msg in st.session_state.messages:
  st.chat_message(msg["role"]).markdown(msg["content"])

query = st.chat_input("Ask anything...")

if query:
  st.session_state.messages.append({"role":"user","content":query})
  st.chat_message("user").markdown(query)

  ans = run_agent(query)

  st.chat_message("ai").markdown(ans)
  st.session_state.messages.append({"role":"ai","content":ans})
