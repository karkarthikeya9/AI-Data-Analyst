import os

import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from groq import Groq

from pandasai.config import ConfigManager
from pandasai_litellm.litellm import LiteLLM

from tools.data_analysis import create_data_agent
from tools import create_tools

from agent import create_agent, run_agent


# ----------------------------
# PAGE CONFIGURATION
# ----------------------------

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="centered",
)


# ----------------------------
# LOAD ENVIRONMENT
# ----------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


# ----------------------------
# CREATE GROQ CLIENT
# ----------------------------

client = Groq(
    api_key=api_key
)


# ----------------------------
# CREATE PANDASAI LLM
# ----------------------------

pandasai_llm = LiteLLM(
    model="groq/openai/gpt-oss-20b"
)


ConfigManager.set(
    {
        "llm": pandasai_llm
    }
)


# ----------------------------
# TITLE
# ----------------------------

st.title("📊 AI Data Analyst")

st.write(
    "Upload a dataset and ask questions "
    "using natural language."
)


st.divider()


# ----------------------------
# FILE UPLOAD
# ----------------------------

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv"],
    help="Drag and drop a CSV file here.",
)


# ----------------------------
# WAIT FOR DATASET
# ----------------------------

if uploaded_file is None:

    st.info(
        "Upload a CSV file to start analyzing your data."
    )

    st.stop()


# ----------------------------
# LOAD DATASET
# ----------------------------

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(
        f"Could not read the dataset: {e}"
    )

    st.stop()


# ----------------------------
# DATASET INFORMATION
# ----------------------------

st.success(
    f"Uploaded: {uploaded_file.name}"
)

st.caption(
    f"{len(df)} rows × {len(df.columns)} columns"
)


with st.expander("Preview dataset"):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ----------------------------
# CREATE PANDASAI AGENT
# ----------------------------

data_agent = create_data_agent(df)


# ----------------------------
# CREATE TOOLS
# ----------------------------

TOOLS = create_tools(data_agent)


# ----------------------------
# CREATE / RESET OUTER AGENT
# ----------------------------

if (
    "agent" not in st.session_state
    or st.session_state.get("filename") != uploaded_file.name
):

    st.session_state.agent = create_agent(
        client,
        TOOLS
    )

    st.session_state.filename = uploaded_file.name


agent = st.session_state.agent


# ----------------------------
# CHAT HISTORY DISPLAY
# ----------------------------

for message in agent["messages"]:

    if message["role"] == "user":

        with st.chat_message("user"):

            st.write(
                message["content"]
            )

    elif message["role"] == "assistant":

        with st.chat_message("assistant"):

            st.write(
                message["content"]
            )


# ----------------------------
# CHAT INPUT
# ----------------------------

question = st.chat_input(
    "Ask a question about your dataset..."
)


if question:

    with st.chat_message("user"):

        st.write(question)


    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing..."
        ):

            response = run_agent(
                agent,
                question
            )

        st.write(response)