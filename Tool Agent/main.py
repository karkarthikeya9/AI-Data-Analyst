import os
import json
import pandas as pd

from dotenv import load_dotenv
from groq import Groq

from pandasai import Agent
from pandasai.config import ConfigManager
from pandasai_litellm.litellm import LiteLLM


# Load environment variables from .env
load_dotenv()

# Get the Groq API key
api_key = os.getenv("GROQ_API_KEY")

# Create the Groq client
client = Groq(api_key=api_key)


# Create the LLM used by PandasAI
pandasai_llm = LiteLLM(
    model="groq/openai/gpt-oss-20b"
)

# Configure PandasAI globally
ConfigManager.set(
    {
        "llm": pandasai_llm
    }
)

# ----------------------------
# OUR DATA TOOL
# ----------------------------

# ----------------------------
# LOAD DATASET
# ----------------------------

df = pd.read_csv("data.csv")


# ----------------------------
# CREATE PANDASAI AGENT
# ----------------------------

data_agent = Agent(df)


# ----------------------------
# OUR DATA TOOL
# ----------------------------

def analyze_data(question):
    result = data_agent.chat(question)

    return str(result)

# ----------------------------
# TOOL DEFINITION
# ----------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "analyze_data",
            "description": (
                "Analyze the local data.csv dataset. "
                "Use this tool whenever the user asks a question about "
                "the dataset, rows, columns, values, sales, profit, "
                "countries, or products."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's question about data.csv"
                    }
                },
                "required": ["question"]
            }
        }
    }
]


# ----------------------------
# AGENT INSTRUCTIONS
# ----------------------------

SYSTEM_PROMPT = """
You are a helpful AI Data Agent.

You can:
- Answer general questions directly.
- Help users understand data.
- Use the analyze_data tool when the user asks
  about the local data.csv dataset.

Be helpful, clear, and concise.
"""


# ----------------------------
# CONVERSATION MEMORY
# ----------------------------

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]


# ----------------------------
# AGENT LOOP
# ----------------------------

while True:

    user_input = input("\nYou: ")

    # Exit condition
    if user_input.lower() in ["exit", "quit"]:
        print("Agent: Goodbye! 👋")
        break

    # Add user's message to conversation history
    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Send conversation to the model
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        tools=tools
    )

    print("\n--- First response received ---")

    response_message = completion.choices[0].message

    print("Content:", response_message.content)
    print("Tool calls:", response_message.tool_calls)

    # Check if the model wants to use a tool
    if response_message.tool_calls:

        print("\n--- Agent wants to use a tool ---")

        # Add the assistant's tool-call message
        messages.append(response_message)

        # Go through each tool call
        for tool_call in response_message.tool_calls:

            tool_name = tool_call.function.name

            print("Tool name:", tool_name )

            arguments = json.loads(
                tool_call.function.arguments
            )

            print("Arguments:", arguments)

            # Run our data analysis tool
            if tool_name == "analyze_data":

                print("\n--- Running analyze_data() ---")

                result = analyze_data(
                    arguments["question"]
                )

                print("Tool result:")
                print(result)

            # Add tool result to conversation
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                }
            )

        print("\n--- Sending tool result back to model ---")

        # Ask the model again with the tool result
        final_completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            tools=tools
        )

        print("\n--- Final response received ---")

        agent_response = (
            final_completion
            .choices[0]
            .message
            .content
        )

    else:

        print("\n--- Normal response ---")

        agent_response = response_message.content


    # Print the agent's response
    print(f"\nAgent: {agent_response}")

    # Save final response to conversation history
    messages.append(
        {
            "role": "assistant",
            "content": agent_response
        }
    )