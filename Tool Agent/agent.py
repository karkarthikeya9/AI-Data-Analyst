import json

from groq import Groq


MODEL = "openai/gpt-oss-20b"

MAX_TOOL_STEPS = 5


# ----------------------------
# TOOL SCHEMAS
# ----------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_data",
            "description": (
                "Analyze the active dataset. "
                "Use this tool whenever information "
                "must be obtained from the dataset."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "A self-contained question about "
                            "the dataset."
                        ),
                    }
                },
                "required": ["question"],
            },
        },
    }
]


# ----------------------------
# SYSTEM PROMPT
# ----------------------------

SYSTEM_PROMPT = """
You are a helpful AI Data Agent.

You have access to an active dataset through
the analyze_data tool.

You are responsible for reasoning through the
user's entire request.

IMPORTANT:

A user's question may require multiple reasoning
steps.

Do not assume that one tool call is enough.

Use the conversation history to resolve references
and previously discovered information.

Examples of references include:

- "that country"
- "that product"
- "the second highest"
- "the lowest sales country"
- "the country with the highest sales"
- "the one we just discussed"

If a question combines dataset information with
general knowledge, break the problem into steps.

For example:

User:
"What is the most popular sport in the least
sales country?"

You should reason:

1. Find the country with the least sales.
2. Once the country is known, determine its
   most popular sport.
3. The second step may be answered using general
   knowledge and does not necessarily require
   the dataset.

Use analyze_data only when information must be
obtained from the dataset.

After receiving a tool result, reassess the
original user question.

If more information is required, use another
tool call.

If you have enough information, provide the
final answer.

Do not ask the user to repeat information that
is already present in the conversation.

Be helpful, clear, and concise.
"""


# ----------------------------
# CREATE AGENT
# ----------------------------

def create_agent(client, tools):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    return {
        "client": client,
        "tools": tools,
        "messages": messages,
    }


# ----------------------------
# RUN AGENT
# ----------------------------

def run_agent(agent, user_input):

    messages = agent["messages"]
    client = agent["client"]
    tools = agent["tools"]


    # ----------------------------
    # ADD USER MESSAGE
    # ----------------------------

    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )


    # ----------------------------
    # REASONING / TOOL LOOP
    # ----------------------------

    for step in range(MAX_TOOL_STEPS):

        print(
            f"\n--- Agent reasoning step "
            f"{step + 1}/{MAX_TOOL_STEPS} ---"
        )


        # ----------------------------
        # ASK GPT-OSS-20B
        # ----------------------------

        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )

        response_message = completion.choices[0].message


        print(
            "Content:",
            response_message.content
        )

        print(
            "Tool calls:",
            response_message.tool_calls
        )


        # ----------------------------
        # NO TOOL REQUIRED
        # ----------------------------

        if not response_message.tool_calls:

            agent_response = response_message.content

            messages.append(
                {
                    "role": "assistant",
                    "content": agent_response,
                }
            )

            return agent_response


        # ----------------------------
        # SAVE ASSISTANT TOOL CALL
        # ----------------------------

        messages.append(response_message)


        # ----------------------------
        # EXECUTE TOOL CALLS
        # ----------------------------

        for tool_call in response_message.tool_calls:

            tool_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )


            print(
                f"\n--- Tool requested: "
                f"{tool_name} ---"
            )

            print(
                "Arguments:",
                arguments
            )


            tool_function = tools.get(
                tool_name
            )


            if tool_function is None:

                result = (
                    f"Unknown tool: {tool_name}"
                )

            else:

                print(
                    f"\n--- Running "
                    f"{tool_name}() ---"
                )

                result = tool_function(
                    arguments["question"]
                )


            print(
                "Tool result:",
                result
            )


            # ----------------------------
            # ADD TOOL RESULT TO MEMORY
            # ----------------------------

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )


        # ----------------------------
        # LOOP CONTINUES
        # ----------------------------

        print(
            "\n--- Tool result added to "
            "conversation. Continuing reasoning. ---"
        )


    # ----------------------------
    # MAXIMUM STEPS REACHED
    # ----------------------------

    fallback_message = (
        "I wasn't able to complete the analysis "
        "within the allowed reasoning steps."
    )

    messages.append(
        {
            "role": "assistant",
            "content": fallback_message,
        }
    )

    return fallback_message 