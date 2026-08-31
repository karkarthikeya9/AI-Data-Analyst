# 🤖 AI Data Analyst

A learning-driven project built while exploring **Groq, GPT-OSS-20B,
tool calling, Pandas, PandasAI, agent loops, conversational context,
persistent chat storage, and Streamlit UI**.

This repository is both a working AI Data Analyst and a **learning
record** documenting how the project evolved from a simple Groq API call
into a multi-component agentic data-analysis application.

------------------------------------------------------------------------

# 📚 GROQ Learning Journey

### Completed

-   [x] 1. Project setup
-   [x] 2. API keys and `.env`
-   [x] 3. Groq Python SDK
-   [x] 4. Create a client
-   [x] 5. Send first message
-   [x] 6. Receive AI response
-   [x] 7. Understand the response object

### Next

-   [ ] Messages and conversation history
-   [ ] System prompts
-   [ ] Model parameters
-   [ ] Streaming responses
-   [ ] Error handling
-   [ ] Structured outputs
-   [ ] Tool/function calling

### Mini Project 2

**Build an actual AI assistant**

This became the bridge from learning the Groq SDK to building an agent
capable of deciding when it needs external tools.

------------------------------------------------------------------------

# 📊 Pandas + PandasAI

The next stage introduced data analysis.

The goal was to combine:

-   Pandas for data manipulation
-   PandasAI for natural-language data analysis
-   Groq for language-model reasoning
-   Tool/function calling for agent behavior

The combined project became:

``` text
User → AI → decides analysis → Pandas/Data tools → Results → AI explanation
```

------------------------------------------------------------------------

# 🧠 Project Evolution

## Phase 1: GPT-OSS-20B Tool-Calling Agent

The first functional agent used GPT-OSS-20B to decide whether a
data-analysis tool should be called.

``` text
GPT-OSS-20B
     ↓
tool decision
     ↓
analyze_data()
     ↓
data.csv
```

The key lesson was that the model does not need to perform every
operation itself. It can decide when software tools are required.

------------------------------------------------------------------------

## Phase 2: PandasAI Fork

The dummy data tool was replaced with the project's fork of PandasAI.

``` text
GPT-OSS-20B
     ↓
analyze_data(question)
     ↓
PandasAI fork
     ↓
Groq
     ↓
generated analysis
     ↓
Pandas
     ↓
data.csv
     ↓
result
```

### Groq vs LiteLLM

**Groq is the actual model provider.**

The PandasAI LiteLLM extension is an adapter that allows PandasAI to
communicate with the Groq-backed model.

``` text
PandasAI
    ↓
pandasai_litellm
    ↓
LiteLLM adapter
    ↓
Groq API
    ↓
GPT-OSS-20B
```

The outer agent can communicate with Groq directly through the Groq SDK,
while PandasAI uses its LLM interface through the adapter.

------------------------------------------------------------------------

## Phase 3: Separate Agent from Data Tool

A major architectural improvement was separating:

1.  The outer agent
2.  The data-analysis tool
3.  The PandasAI data agent

The goal was to stop creating a new PandasAI agent for every question
and instead maintain the appropriate analysis context for the dataset.

``` text
GPT-OSS-20B
     ↓
Tool system
     ↓
analyze_data()
     ↓
PandasAI
```

------------------------------------------------------------------------

## Phase 4: Cleaner Tool System

The tool architecture evolved toward multiple specialized tools:

``` text
                    GPT-OSS-20B
                         │
             ┌───────────┼───────────┐
             ↓           ↓           ↓
       analyze_data  dataset_info  create_chart
             │
             ↓
         PandasAI
```

The goal is for the outer agent to decide which capability is
appropriate instead of putting every responsibility into one function.

------------------------------------------------------------------------

## Phase 5: Different Result Types

The system also needs to handle different types of analytical output.

``` text
                  PandasAI
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
      number      DataFrame      chart
        │            │            │
        └────────────┼────────────┘
                     ↓
                  Agent
                     ↓
               User response
```

------------------------------------------------------------------------

# 🔁 Agent Reasoning Loop

The project moved beyond a simple:

``` text
Question → Tool → Answer
```

architecture.

The agent was redesigned around a reasoning loop:

``` text
                 User
                  │
                  ▼
             Agent Loop
                  │
                  ▼
             GPT-OSS-20B
                  │
        ┌─────────┴─────────┐
        │                   │
    Tool needed          No tool
        │                   │
        ▼                   ▼
   analyze_data          Answer
        │
        ▼
     PandasAI
        │
        ▼
      Result
        │
        └──────────────┐
                       │
                       ▼
                 GPT-OSS-20B
                       │
                  Enough info?
                  /          \
                No            Yes
                │              │
                ▼              ▼
             Tool again      Answer
```

This allows the model to inspect a tool result and decide whether
another tool call is necessary.

------------------------------------------------------------------------

# 🧠 Conversational Context

One of the harder problems was resolving references across turns.

For example:

``` text
User:
What about the second highest?

Agent:
The country with the second highest sales is India.

User:
Popular sport in that country

Agent:
The most popular sport in India is cricket.
```

The challenge becomes:

``` text
"that country"
      ↓
entity established by previous conversation
      ↓
India
```

The project therefore moved toward keeping conversational context at the
**outer agent level**, rather than relying only on context inside the
PandasAI analysis loop.

------------------------------------------------------------------------

# 🔄 Outer Agent + PandasAI Loop

The desired architecture became:

``` text
                    USER
                      │
                      ▼
               Outer Agent
                      │
                      ▼
                 GPT-OSS-20B
                      │
             ┌────────┴────────┐
             │                 │
        Tool needed         No tool
             │                 │
             ▼                 ▼
        PandasAI             Answer
             │
             ▼
          Result
             │
             ▼
        GPT-OSS-20B
             │
        Enough context?
          /       \
        No         Yes
        │           │
        ▼           ▼
   Tool again     Answer
```

The important idea is that **Groq and PandasAI cooperate in an outer
loop until the agent has enough information to produce the desired
response.**

------------------------------------------------------------------------

# 🔍 Query Optimization: Future Question

A future idea was to use Groq to rewrite a user's natural-language
question before sending it to PandasAI.

``` text
User question
      ↓
Groq query optimizer
      ↓
PandasAI
```

The concern is whether this would simply waste tokens because Groq is
already involved elsewhere.

Potential downsides:

-   Additional tokens
-   Additional latency
-   Another failure point
-   Possible distortion of the user's intent

Therefore, query optimization is currently treated as a **future
experiment**, not a mandatory extra LLM call.

The outer agent already interprets natural-language questions and
formulates tool calls.

------------------------------------------------------------------------

# 🖥️ Frontend / UI Journey

The project then evolved from a terminal-based agent into a Streamlit
application.

## Original UI roadmap

### V4.1

-   [x] Frontend UI

### V4.2

-   [x] Frontend → backend

### V4.3

-   [x] Drag/drop → DataFrame

### V4.4

-   [x] DataFrame → PandasAI  

### V4.5

-   [x] Chat history

### V4.6

-   [x] Charts / results

------------------------------------------------------------------------

# 🚀 V5 Roadmap

  -----------------------------------------------------------------------
  Version                 Feature                 Goal
  ----------------------- ----------------------- -----------------------
  **V5.2**                Persistent dataset      Uploaded dataset
                                                  survives Streamlit
                                                  reruns

  **V5.3**                Chat history            Proper conversational
                                                  UI + follow-ups

  **V5.4**                CSV + Excel             `.csv`, `.xlsx`, `.xls`
                                                  uploads

  **V5.5**                Clear / New Dataset     Reset dataset and
                                                  conversation cleanly

  **V5.6**                Charts                  Let the agent
                                                  return/display
                                                  visualizations

  **V5.7**                Error handling          Friendly failures
                                                  instead of ugly
                                                  tracebacks

  **V5.8**                UI foundation           Clean components that
                                                  can later be styled to
                                                  the final design
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 💬 Chat-Specific Dataset Architecture

A key product decision was:

> **Each dataset belongs to its chat.**

The intended structure is:

``` text
AI DATA ANALYST
│
├── 💬 Chat 1
│    ├── 📄 sales.csv
│    ├── 🤖 Agent
│    └── 💬 Conversation
│
├── 💬 Chat 2
│    ├── 📄 customers.xlsx
│    ├── 🤖 Agent
│    └── 💬 Conversation
│
└── 💬 Chat 3
     ├── 📄 products.csv
     ├── 🤖 Agent
     └── 💬 Conversation
```

Each conversation has its own analytical environment.

This allows users to:

-   Create multiple chats
-   Switch between chats
-   Continue previous conversations
-   Keep datasets isolated
-   Delete individual chats
-   Rename chats
-   Preserve other chats when one is deleted

------------------------------------------------------------------------

# 💾 Chat and Dataset Storage

The current architecture conceptually stores:

``` text
Chat
│
├── Chat ID
├── Title
├── Dataset filename
├── Dataset path
├── Created time
└── Updated time
        │
        ▼
Messages
│
├── User message
├── Assistant message
├── User message
└── Assistant message
```

The dataset and conversation history are associated with the chat.

The current local implementation uses SQLite for chat/history metadata
and local files for uploaded datasets.

------------------------------------------------------------------------

# 📄 Dataset Support

The application evolved from a fixed `data.csv` workflow into uploaded
datasets.

Supported formats:

``` text
.csv
.xlsx
.xls
```

The intended flow is:

``` text
Drag & Drop Dataset
        ↓
    DataFrame
        ↓
    Create chat
        ↓
    PandasAI
        ↓
   Ask questions
```

------------------------------------------------------------------------

# 📊 Visualization System

Charts were introduced so that analytical responses can contain both:

``` text
Answer
+
Visualization
```

The visualization architecture separates **reasoning** from
**rendering**.
 
``` text
User question
      ↓
Groq / Agent
      ↓
Should visualization help?
      ↓
Chart plan
      ↓
Python / Pandas
      ↓
Actual data
      ↓
Streamlit chart
```

The LLM decides what visualization is useful, while Python performs the
actual data operation and rendering.

This helps prevent the model from inventing numerical results.

### Current visualization direction

-   Bar charts
-   Line charts
-   Scatter plots
-   Automatic visualization planning
-   Safe handling of invalid X/Y selections

For example, if:

``` text
X = Sales
Y = Sales
```

the UI should show a helpful message instead of crashing.

------------------------------------------------------------------------

# 🧪 Example Interactions

### Data question

``` text
User:
Which country has the highest total sales?
```

``` text
Agent:
The country with the highest total sales is USA.
```

### Follow-up

``` text
User:
What about the second highest?
```

``` text
Agent:
The second highest country by total sales is India.
```

### Contextual question

``` text
User:
Popular sport in that country
```

The outer agent should resolve:

``` text
"that country"
      ↓
India
```

and answer using the appropriate source of knowledge.

### Multi-step question

``` text
User:
What is the 4th sales country and what is the most favourite food
of that country and the standard list of ingredients used in that dish?
```

The agent can first use the dataset to determine the fourth-ranked
country and then continue with the non-dataset part of the question.

------------------------------------------------------------------------

# 🧱 Current High-Level Architecture

``` text
                         USER
                           │
                           ▼
                    Streamlit UI
                           │
                           ▼
                    Conversation
                           │
                           ▼
                   Outer AI Agent
                           │
                           ▼
                      GPT-OSS-20B
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
        analyze_data  dataset_info  create_chart
              │
              ▼
          PandasAI
              │
              ▼
        Pandas / DataFrame
              │
              ▼
           Result
              │
              ▼
        Outer Agent
              │
       ┌──────┴──────┐
       ▼             ▼
    Answer        Visualization
       │             │
       └──────┬──────┘
              ▼
             USER
```

------------------------------------------------------------------------

# 🧩 Technology Stack

## AI / LLM

-   Groq
-   GPT-OSS-20B
-   Groq Python SDK
-   PandasAI
-   PandasAI LiteLLM extension

## Data

-   Python
-   Pandas
-   PandasAI
-   CSV
-   XLSX
-   XLS

## Agent

-   Python
-   Groq tool/function calling
-   Agent reasoning loop
-   Tool registry
-   Conversation context

## Frontend

-   Streamlit
-   Drag-and-drop dataset upload
-   Multi-chat UI
-   Dataset preview
-   Chat history
-   Charts

## Storage

-   SQLite for current local chat/history architecture
-   Per-chat dataset storage

------------------------------------------------------------------------

# 📁 Project Structure

``` text
AI-Data-Analyst/
│
├── README.md
│
├── Tool Agent/
│   ├── frontend.py
│   ├── main.py
│   ├── agent.py
│   ├── storage.py
│   ├── tools/
│   │   ├── __init__.py
│   │   └── data_analysis.py
│   └── uploads/
│
├── pandas-ai/
│   ├── pandasai/
│   ├── extensions/
│   │   └── llms/
│   │       └── litellm/
│   │           └── pandasai_litellm/
│   └── pyproject.toml
│
└── exports/
```

------------------------------------------------------------------------

# 🔐 Environment Variables

The project uses environment variables for secrets.

Example:

``` text
GROQ_API_KEY=your_api_key_here
```

The API key should be kept in `.env` during local development and should
never be committed to GitHub.

------------------------------------------------------------------------

# 🎓 Development Philosophy

This project is intentionally being developed as a **learning record**,
not simply as a finished application.

Each stage answers a different engineering question:

``` text
Groq
 ↓
How do I communicate with an LLM?

Tool calling
 ↓
How does an LLM decide to use software?

Pandas
 ↓
How do I manipulate real data?

PandasAI
 ↓
How can natural language control data analysis?

Agent loop
 ↓
How can the model decide whether it needs more information?

Conversation memory
 ↓
How can the system understand follow-up questions?

Chat-specific datasets
 ↓
How do I isolate different analytical sessions?

Streamlit
 ↓
How do I turn the agent into a usable application?

Charts
 ↓
How can the agent communicate analytical results visually?
```

------------------------------------------------------------------------

# 🛣️ Future Roadmap

## Agent Intelligence

-   [ ] Better contextual reference resolution
-   [ ] More reliable multi-step reasoning
-   [ ] Better tool selection
-   [ ] More specialized data tools
-   [ ] Structured tool outputs
-   [ ] Better distinction between data questions and general questions

## Dataset Experience

-   [ ] Better dataset information panel
-   [ ] Column/type summaries
-   [ ] Dataset replacement controls
-   [ ] Dataset removal controls
-   [ ] Larger dataset handling
-   [ ] More file formats where useful

## Visualization

-   [ ] Automatically generated chart titles
-   [ ] More chart types
-   [ ] Better formatting
-   [ ] Interactive chart controls
-   [ ] Chart download
-   [ ] Visualization history
-   [ ] Better AI-generated visualization recommendations

## UI

-   [ ] Final custom UI design
-   [ ] More polished components
-   [ ] Improved responsive layout
-   [ ] Better loading states
-   [ ] Improved error presentation

## Production Storage

The current local architecture uses SQLite and local dataset files.

For a production deployment, persistent cloud storage should replace
reliance on the application's local filesystem.

Future architecture:

``` text
                         Streamlit
                            │
              ┌─────────────┴─────────────┐
              │                           │
        Chat history                 Datasets
              │                           │
              ▼                           ▼
      Persistent database          Persistent storage
```

The goal is for users to return later and find:

``` text
Chat A
 ├── sales.csv
 └── conversation history

Chat B
 ├── customers.xlsx
 └── conversation history

Chat C
 ├── products.csv
 └── conversation history
```

------------------------------------------------------------------------

# 🏁 Final Product Vision

The long-term goal is an AI Data Analyst where a user does not need to
know Pandas, SQL, chart libraries, or data-analysis syntax.

They provide a dataset and ask questions naturally.

``` text
                 AI DATA ANALYST
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼
    Dataset          Agent          Conversation
       │               │               │
       │               ▼               │
       │          GPT-OSS-20B          │
       │               │               │
       │        ┌──────┴──────┐        │
       │        ▼             ▼        │
       │     PandasAI      General     │
       │        │          knowledge   │
       │        ▼                       │
       │      Pandas                    │
       │        │                       │
       └────────┼───────────────────────┘
                ▼
          Answer / Table /
          Visualization
                │
                ▼
               USER
```

Each conversation remains isolated:

``` text
AI DATA ANALYST
│
├── 💬 Chat 1
│    ├── 📄 sales.csv
│    ├── 🤖 Agent
│    └── 💬 Conversation
│
├── 💬 Chat 2
│    ├── 📄 customers.xlsx
│    ├── 🤖 Agent
│    └── 💬 Conversation
│
└── 💬 Chat 3
     ├── 📄 products.csv
     ├── 🤖 Agent
     └── 💬 Conversation
```

------------------------------------------------------------------------

# 📖 Learning Record

This project started as a way to learn **Groq and LLM fundamentals**.

It gradually became a practical exploration of:

-   LLM APIs
-   SDKs
-   Tool/function calling
-   Agent architecture
-   Reasoning loops
-   Conversation context
-   Pandas
-   PandasAI
-   LLM adapters
-   Dataset isolation
-   Persistent storage
-   Streamlit
-   Visualization
-   Frontend architecture

The most important part is not only the final AI Data Analyst.

It is the progression:

``` text
Learning
   ↓
Experiment
   ↓
Mini projects
   ↓
Tool calling
   ↓
Data analysis
   ↓
Agent
   ↓
Memory
   ↓
Multi-chat architecture
   ↓
Frontend
   ↓
Visualization
   ↓
AI Data Analyst
```

**This README is intended to remain a living learning record as the
project continues to evolve.**
