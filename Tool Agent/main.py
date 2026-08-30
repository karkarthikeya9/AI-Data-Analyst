import os
import pandas as pd

from dotenv import load_dotenv
from groq import Groq

from pandasai.config import ConfigManager
from pandasai_litellm.litellm import LiteLLM

from tools.data_analysis import create_data_agent
from tools import create_tools

from agent import create_agent, run_agent


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


# ----------------------------
# CONFIGURE PANDASAI
# ----------------------------

ConfigManager.set(
    {
        "llm": pandasai_llm
    }
)


# ----------------------------
# LOAD DATASET
# ----------------------------

df = pd.read_csv("data.csv")


# ----------------------------
# CREATE PANDASAI AGENT
# ----------------------------

data_agent = create_data_agent(df)


# ----------------------------
# CREATE TOOL REGISTRY
# ----------------------------

TOOLS = create_tools(data_agent)


# ----------------------------
# CREATE AI AGENT
# ----------------------------

agent = create_agent(
    client,
    TOOLS
)


# ----------------------------
# CLI LOOP
# ----------------------------

while True:

    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]:

        print("Agent: Goodbye! 👋")

        break

    response = run_agent(
        agent,
        user_input
    )

    print(f"\nAgent: {response}")