import pandas as pd
import streamlit as st

from pandasai import Agent
from pandasai.config import ConfigManager
from pandasai_litellm.litellm import LiteLLM


# ----------------------------
# PAGE CONFIGURATION
# ----------------------------

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="centered",
)


# ----------------------------
# PANDASAI CONFIGURATION
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
# UI
# ----------------------------

st.title("📊 AI Data Analyst")

st.write(
    "Upload a dataset and ask questions about your data."
)

st.divider()


uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv"],
    help="Drag and drop a CSV file here, or click Browse files.",
)


# ----------------------------
# DATASET
# ----------------------------

if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    try:

        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.caption(
            f"{len(df)} rows × {len(df.columns)} columns"
        )

        # Create PandasAI agent for uploaded dataset
        data_agent = Agent(df)

        st.divider()

        st.subheader("Ask your data")

        question = st.chat_input(
            "Ask a question about your dataset..."
        )

        if question:

            st.write(f"**You:** {question}")

            with st.spinner("Analyzing your data..."):

                result = data_agent.chat(question)

            st.write(f"**Agent:** {result}")

    except Exception as e:

        st.error(
            f"Could not process the dataset: {e}"
        )