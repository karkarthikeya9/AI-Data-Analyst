import os
import json

import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from groq import Groq

from pandasai.config import ConfigManager
from pandasai_litellm.litellm import LiteLLM

from tools.data_analysis import create_data_agent
from tools import create_tools

from agent import create_agent, run_agent

import storage


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       GLOBAL
    ------------------------------------------------------- */

    .stApp {
        background: #0b0d12;
    }

    [data-testid="stSidebar"] {
        background: #171922;
        border-right: 1px solid #292c36;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }


    /* -------------------------------------------------------
       TITLES
    ------------------------------------------------------- */

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #8d93a1;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }


    /* -------------------------------------------------------
       DATASET CARD
    ------------------------------------------------------- */

    .dataset-card {
        background: #151821;
        border: 1px solid #292d38;
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 20px;
    }

    .dataset-name {
        font-size: 1.05rem;
        font-weight: 600;
    }

    .dataset-meta {
        color: #8d93a1;
        font-size: 0.88rem;
        margin-top: 4px;
    }


    /* -------------------------------------------------------
       CHAT
    ------------------------------------------------------- */

    [data-testid="stChatMessage"] {
        background: transparent;
    }


    /* -------------------------------------------------------
       BUTTONS
    ------------------------------------------------------- */

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #303440;
        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        border-color: #777f91;
    }


    /* -------------------------------------------------------
       TABS
    ------------------------------------------------------- */

    button[data-baseweb="tab"] {
        font-size: 0.95rem;
    }


    /* -------------------------------------------------------
       FILE UPLOADER
    ------------------------------------------------------- */

    [data-testid="stFileUploader"] {
        background: #151821;
        border-radius: 14px;
        padding: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# STORAGE
# ============================================================

storage.initialize_database()


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:

    st.error(
        "GROQ_API_KEY is not configured."
    )

    st.stop()


# ============================================================
# GROQ
# ============================================================

client = Groq(
    api_key=api_key
)


# ============================================================
# PANDASAI
# ============================================================

pandasai_llm = LiteLLM(
    model="groq/openai/gpt-oss-20b"
)

ConfigManager.set(
    {
        "llm": pandasai_llm
    }
)


# ============================================================
# SESSION STATE
# ============================================================

if "active_chat_id" not in st.session_state:

    st.session_state.active_chat_id = None


if "agents" not in st.session_state:

    st.session_state.agents = {}


if "renaming_chat_id" not in st.session_state:

    st.session_state.renaming_chat_id = None


# ============================================================
# DATASET LOADER
# ============================================================

def load_dataset(file_path):

    extension = (
        os.path.splitext(file_path)[1]
        .lower()
    )


    if extension == ".csv":

        return pd.read_csv(
            file_path
        )


    if extension == ".xlsx":

        return pd.read_excel(
            file_path,
            engine="openpyxl"
        )


    if extension == ".xls":

        return pd.read_excel(
            file_path,
            engine="xlrd"
        )


    raise ValueError(
        f"Unsupported file type: {extension}"
    )


# ============================================================
# CREATE / LOAD AGENT
# ============================================================

def get_agent(
    chat_id,
    df
):

    if chat_id not in st.session_state.agents:

        data_agent = create_data_agent(
            df
        )

        tools = create_tools(
            data_agent
        )

        agent = create_agent(
            client,
            tools
        )


        # -----------------------------------------------
        # Restore conversation
        # -----------------------------------------------

        saved_messages = storage.get_messages(
            chat_id
        )


        for message in saved_messages:

            agent["messages"].append(
                {
                    "role": message["role"],
                    "content": message["content"],
                }
            )


        st.session_state.agents[
            chat_id
        ] = agent


    return st.session_state.agents[
        chat_id
    ]


# ============================================================
# AI VISUALIZATION PLANNER
# ============================================================

def get_visualization_plan(
    question,
    df
):

    columns = list(
        df.columns
    )

    numeric_columns = list(
        df.select_dtypes(
            include="number"
        ).columns
    )

    categorical_columns = list(
        df.select_dtypes(
            exclude="number"
        ).columns
    )


    prompt = f"""
You are the visualization planner for an AI Data Analyst.

The user asked:

{question}

The dataset contains these columns:

{columns}

Numeric columns:

{numeric_columns}

Categorical/text columns:

{categorical_columns}

Decide whether a visualization would materially improve
the answer.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "visualize": true,
    "chart_type": "bar",
    "x_column": "Country",
    "y_column": "Sales",
    "reason": "Sales should be compared across countries."
}}

Rules:

1. visualize must be true or false.

2. chart_type must be one of:
   - bar
   - line
   - scatter
   - none

3. x_column must be an actual dataset column.

4. y_column must be an actual dataset column.

5. For bar charts:
   - Prefer categorical columns for x_column.
   - Prefer numeric columns for y_column.

6. For line charts:
   - Prefer time/order columns for x_column.
   - Prefer numeric columns for y_column.

7. For scatter charts:
   - Both x_column and y_column must be numeric.
   - They must be DIFFERENT columns.

8. If a chart would not help, return:

{{
    "visualize": false,
    "chart_type": "none",
    "x_column": null,
    "y_column": null,
    "reason": "A visualization is not useful for this question."
}}

9. Never invent column names.

Return JSON only.
"""


    try:

        completion = client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON "
                        "visualization planner."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0

        )


        content = (
            completion
            .choices[0]
            .message
            .content
        )


        plan = json.loads(
            content
        )


        return plan


    except Exception as e:

        print(
            "Visualization planner error:",
            e
        )

        return {
            "visualize": False,
            "chart_type": "none",
            "x_column": None,
            "y_column": None,
            "reason": "Visualization planning failed."
        }

# ============================================================
# RENDER AI VISUALIZATION
# ============================================================

def render_ai_visualization(
    plan,
    df
):

    if not plan:

        return


    if not plan.get(
        "visualize",
        False
    ):

        return


    chart_type = plan.get(
        "chart_type"
    )

    x_column = plan.get(
        "x_column"
    )

    y_column = plan.get(
        "y_column"
    )


    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    if (
        x_column not in df.columns
        or y_column not in df.columns
    ):

        return


    # --------------------------------------------------------
    # BAR
    # --------------------------------------------------------

    if chart_type == "bar":

        try:

            chart_df = (
                df
                .groupby(
                    x_column,
                    dropna=False
                )[y_column]
                .sum()
                .sort_values(
                    ascending=False
                )
                .head(20)
                .reset_index()
            )


            if chart_df.empty:

                return


            st.markdown(
                "#### 📊 Visualization"
            )


            st.bar_chart(
                chart_df.set_index(
                    x_column
                )[y_column],
                use_container_width=True
            )


        except Exception as e:

            st.caption(
                f"Could not create visualization: {e}"
            )


    # --------------------------------------------------------
    # LINE
    # --------------------------------------------------------

    elif chart_type == "line":

        try:

            chart_df = df[
                [
                    x_column,
                    y_column
                ]
            ].dropna()


            if chart_df.empty:

                return


            chart_df = chart_df.head(
                100
            )


            st.markdown(
                "#### 📈 Visualization"
            )


            st.line_chart(
                chart_df.set_index(
                    x_column
                )[y_column],
                use_container_width=True
            )


        except Exception as e:

            st.caption(
                f"Could not create visualization: {e}"
            )


    # --------------------------------------------------------
    # SCATTER
    # --------------------------------------------------------

    elif chart_type == "scatter":

        # Prevent same-column crash

        if x_column == y_column:

            st.warning(
                f"⚠️ X axis and Y axis are both "
                f"'{x_column}'."
            )

            return


        try:

            chart_df = df[
                [
                    x_column,
                    y_column
                ]
            ].dropna()


            if chart_df.empty:

                return


            st.markdown(
                "#### 🔵 Visualization"
            )


            st.scatter_chart(
                chart_df,
                x=x_column,
                y=y_column,
                use_container_width=True
            )


        except Exception as e:

            st.caption(
                f"Could not create visualization: {e}"
            )

# ============================================================
# DELETE EMPTY CHAT HELPER
# ============================================================

def remove_abandoned_empty_chat(
    current_chat_id,
    target_chat_id
):

    if not current_chat_id:

        return


    if current_chat_id == target_chat_id:

        return


    if storage.is_empty_chat(
        current_chat_id
    ):

        storage.delete_chat(
            current_chat_id
        )


        if current_chat_id in (
            st.session_state.agents
        ):

            del st.session_state.agents[
                current_chat_id
            ]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 💬 Conversations"
    )

    st.caption(
        "Your private analysis sessions"
    )

    st.divider()


    # ========================================================
    # NEW CHAT
    # ========================================================

    chats = storage.get_chats()

    existing_empty_chat = None

    for chat in chats:

        if storage.is_empty_chat(
            chat["id"]
        ):

            existing_empty_chat = chat["id"]
            break


    if st.button(
        "＋  New Chat",
        use_container_width=True
    ):

        if existing_empty_chat:

            st.session_state.active_chat_id = (
                existing_empty_chat
            )

        else:

            chat_id = storage.create_chat(
                title="New Chat"
            )

            st.session_state.active_chat_id = (
                chat_id
            )

        st.rerun()


    st.divider()


    # ========================================================
    # CHAT LIST
    # ========================================================

    chats = storage.get_chats()


    if not chats:

        st.caption(
            "No conversations yet."
        )


    for chat in chats:

        chat_id = chat["id"]

        title = chat["title"]


        row_left, row_right = st.columns(
            [6, 1],
            gap="small"
        )


        with row_left:

            if st.button(
                f"💬  {title}",
                key=f"chat_{chat_id}",
                use_container_width=True
            ):

                remove_abandoned_empty_chat(
                    st.session_state.active_chat_id,
                    chat_id
                )

                st.session_state.active_chat_id = (
                    chat_id
                )

                st.rerun()


        with row_right:

            with st.popover(
                "⋮",
                use_container_width=True
            ):

                if st.button(
                    "✏️ Rename",
                    key=f"rename_{chat_id}",
                    use_container_width=True
                ):

                    st.session_state.renaming_chat_id = (
                        chat_id
                    )

                    st.rerun()


                if st.button(
                    "🗑️ Delete",
                    key=f"delete_{chat_id}",
                    use_container_width=True
                ):

                    storage.delete_chat(
                        chat_id
                    )


                    if chat_id in (
                        st.session_state.agents
                    ):

                        del st.session_state.agents[
                            chat_id
                        ]


                    remaining = (
                        storage.get_chats()
                    )


                    if remaining:

                        st.session_state.active_chat_id = (
                            remaining[0]["id"]
                        )

                    else:

                        st.session_state.active_chat_id = (
                            None
                        )


                    st.rerun()


        # ----------------------------------------------------
        # INLINE RENAME
        # ----------------------------------------------------

        if (
            st.session_state.renaming_chat_id
            == chat_id
        ):

            new_title = st.text_input(
                "Rename",
                value=title,
                key=f"rename_input_{chat_id}"
            )


            save_col, cancel_col = st.columns(
                2
            )


            with save_col:

                if st.button(
                    "Save",
                    key=f"save_{chat_id}",
                    use_container_width=True
                ):

                    new_title = (
                        new_title.strip()
                    )


                    if new_title:

                        storage.rename_chat(
                            chat_id,
                            new_title
                        )


                    st.session_state.renaming_chat_id = (
                        None
                    )

                    st.rerun()


            with cancel_col:

                if st.button(
                    "Cancel",
                    key=f"cancel_{chat_id}",
                    use_container_width=True
                ):

                    st.session_state.renaming_chat_id = (
                        None
                    )

                    st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📊 AI Data Analyst</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions, explore your data, and discover patterns.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# NO ACTIVE CHAT
# ============================================================

if st.session_state.active_chat_id is None:

    st.markdown(
        "### Start a new analysis"
    )

    st.write(
        "Upload a CSV or Excel dataset to create a "
        "new analysis session."
    )


    uploaded_file = st.file_uploader(
        "Drop your dataset here",
        type=["csv", "xlsx", "xls"],
        key="initial_upload"
    )


    if uploaded_file:

        chat_id = storage.create_chat(
            title=uploaded_file.name
        )


        dataset_path = storage.save_dataset(
            chat_id,
            uploaded_file
        )


        try:

            df = load_dataset(
                dataset_path
            )

        except Exception as e:

            storage.delete_chat(
                chat_id
            )

            st.error(
                f"Could not read the dataset: {e}"
            )

            st.stop()


        data_agent = create_data_agent(
            df
        )

        tools = create_tools(
            data_agent
        )


        st.session_state.agents[
            chat_id
        ] = create_agent(
            client,
            tools
        )


        st.session_state.active_chat_id = (
            chat_id
        )

        st.rerun()


    st.stop()


# ============================================================
# ACTIVE CHAT
# ============================================================

chat_id = (
    st.session_state.active_chat_id
)

chat = storage.get_chat(
    chat_id
)


if chat is None:

    st.session_state.active_chat_id = None

    st.rerun()


# ============================================================
# DATASET UPLOAD FOR EMPTY CHAT
# ============================================================

dataset_path = chat["dataset_path"]


if dataset_path is None:

    st.markdown(
        "### Upload a dataset"
    )

    st.info(
        "This chat is waiting for a dataset."
    )


    uploaded_file = st.file_uploader(
        "Drop CSV, XLSX or XLS here",
        type=["csv", "xlsx", "xls"],
        key=f"upload_{chat_id}"
    )


    if uploaded_file:

        dataset_path = storage.save_dataset(
            chat_id,
            uploaded_file
        )


        storage.rename_chat(
            chat_id,
            uploaded_file.name
        )


        st.rerun()


    st.stop()


# ============================================================
# LOAD DATASET
# ============================================================

try:

    df = load_dataset(
        dataset_path
    )

except Exception as e:

    st.error(
        f"Could not load dataset: {e}"
    )

    st.stop()


# ============================================================
# HEADER ACTIONS
# ============================================================

header_left, header_right = st.columns(
    [5, 2]
)


with header_left:

    st.markdown(
        f"### {chat['title']}"
    )

    st.caption(
        f"📄 {chat['dataset_filename']}  •  "
        f"{len(df):,} rows  •  "
        f"{len(df.columns):,} columns"
    )


with header_right:

    if st.button(
        "Rename Chat ✏️",
        use_container_width=True,
        key="header_rename"
    ):

        st.session_state.renaming_chat_id = (
            chat_id
        )

        st.rerun()


    if st.button(
        "Delete Chat 🗑️",
        use_container_width=True,
        key="header_delete"
    ):

        storage.delete_chat(
            chat_id
        )


        if chat_id in st.session_state.agents:

            del st.session_state.agents[
                chat_id
            ]


        remaining = storage.get_chats()


        if remaining:

            st.session_state.active_chat_id = (
                remaining[0]["id"]
            )

        else:

            st.session_state.active_chat_id = (
                None
            )


        st.rerun()


# ============================================================
# HEADER RENAME
# ============================================================

if (
    st.session_state.renaming_chat_id
    == chat_id
):

    rename_left, rename_right = st.columns(
        [5, 1]
    )


    with rename_left:

        new_title = st.text_input(
            "Chat name",
            value=chat["title"],
            key=f"header_rename_input_{chat_id}"
        )


    with rename_right:

        if st.button(
            "Save",
            key=f"header_save_{chat_id}",
            use_container_width=True
        ):

            new_title = (
                new_title.strip()
            )


            if new_title:

                storage.rename_chat(
                    chat_id,
                    new_title
                )


            st.session_state.renaming_chat_id = (
                None
            )

            st.rerun()


# ============================================================
# MAIN TABS
# ============================================================

chat_tab, data_tab, chart_tab = st.tabs(
    [
        "💬 Chat",
        "🗂️ Dataset",
        "📊 Charts",
    ]
)


# ============================================================
# DATASET TAB
# ============================================================

with data_tab:

    st.markdown(
        "### Dataset"
    )


    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    info1, info2, info3, info4 = st.columns(
        4
    )


    with info1:

        st.metric(
            "Rows",
            f"{len(df):,}"
        )


    with info2:

        st.metric(
            "Columns",
            f"{len(df.columns):,}"
        )


    with info3:

        numeric_columns = (
            df.select_dtypes(
                include="number"
            ).columns
        )

        st.metric(
            "Numeric",
            f"{len(numeric_columns):,}"
        )


    with info4:

        st.metric(
            "Missing values",
            f"{int(df.isna().sum().sum()):,}"
        )


    st.divider()


    # --------------------------------------------------------
    # Dataset preview
    # --------------------------------------------------------

    st.markdown(
        "#### Preview"
    )

    st.dataframe(
        df.head(100),
        use_container_width=True,
        height=450
    )


    st.divider()


    # --------------------------------------------------------
    # Column information
    # --------------------------------------------------------

    st.markdown(
        "#### Columns"
    )


    column_info = pd.DataFrame(
        {
            "Column": df.columns,
            "Type": [
                str(dtype)
                for dtype in df.dtypes
            ],
            "Missing": [
                int(df[column].isna().sum())
                for column in df.columns
            ],
            "Unique": [
                int(df[column].nunique())
                for column in df.columns
            ],
        }
    )


    st.dataframe(
        column_info,
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    # --------------------------------------------------------
    # Replace dataset
    # --------------------------------------------------------

    st.markdown(
        "#### Replace dataset"
    )


    st.caption(
        "Replacing the dataset keeps this chat but "
        "changes the data it analyzes."
    )


    replacement_file = st.file_uploader(
        "Upload replacement dataset",
        type=["csv", "xlsx", "xls"],
        key=f"replacement_{chat_id}"
    )


    if replacement_file:

        new_path = storage.save_dataset(
            chat_id,
            replacement_file
        )


        storage.rename_chat(
            chat_id,
            replacement_file.name
        )


        # Important:
        # Rebuild the PandasAI agent because the
        # underlying DataFrame has changed.

        if chat_id in st.session_state.agents:

            del st.session_state.agents[
                chat_id
            ]


        st.success(
            f"Dataset replaced with "
            f"{replacement_file.name}"
        )

        st.rerun()


# ============================================================
# CHART TAB
# ============================================================

with chart_tab:

    st.markdown(
        "### 📊 Data Explorer"
    )

    st.caption(
        "Create quick visualizations directly from "
        "the active dataset."
    )


    columns = list(
        df.columns
    )


    numeric_columns = list(
        df.select_dtypes(
            include="number"
        ).columns
    )


    categorical_columns = list(
        df.select_dtypes(
            exclude="number"
        ).columns
    )


    if not columns:

        st.warning(
            "No columns available."
        )

    else:

        chart_type = st.selectbox(
            "Chart type",
            [
                "Bar",
                "Line",
                "Scatter",
            ],
            key=f"chart_type_{chat_id}"
        )


        if chart_type == "Bar":

            if not categorical_columns:

                st.warning(
                    "A bar chart needs a categorical column."
                )

            elif not numeric_columns:

                st.warning(
                    "A bar chart needs a numeric column."
                )

            else:

                x_column = st.selectbox(
                    "Category",
                    categorical_columns,
                    key=f"bar_x_{chat_id}"
                )


                y_column = st.selectbox(
                    "Value",
                    numeric_columns,
                    key=f"bar_y_{chat_id}"
                )


                chart_df = (
                    df.groupby(
                        x_column,
                        dropna=False
                    )[y_column]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                    .head(20)
                    .reset_index()
                )


                st.bar_chart(
                    chart_df.set_index(
                        x_column
                    )[y_column],
                    use_container_width=True
                )


        elif chart_type == "Line":

            if not numeric_columns:

                st.warning(
                    "A line chart needs a numeric column."
                )

            else:

                x_column = st.selectbox(
                    "X axis",
                    columns,
                    key=f"line_x_{chat_id}"
                )


                y_column = st.selectbox(
                    "Value",
                    numeric_columns,
                    key=f"line_y_{chat_id}"
                )


                line_df = df[
                    [
                        x_column,
                        y_column
                    ]
                ].dropna()


                if not pd.api.types.is_numeric_dtype(
                    line_df[x_column]
                ):

                    line_df = (
                        line_df
                        .groupby(
                            x_column
                        )[y_column]
                        .sum()
                        .reset_index()
                    )


                line_df = line_df.head(
                    100
                )


                st.line_chart(
                    line_df.set_index(
                        x_column
                    )[y_column],
                    use_container_width=True
                )


        elif chart_type == "Scatter":

            if len(numeric_columns) < 2:

                st.warning(
                    "Scatter plots need at least two numeric columns."
                )

            else:

                x_column = st.selectbox(
                    "X axis",
                    numeric_columns,
                    key=f"scatter_x_{chat_id}"
                )

                y_column = st.selectbox(
                    "Y axis",
                    numeric_columns,
                    key=f"scatter_y_{chat_id}"
                )

                # ----------------------------------------------------
                # SAME COLUMN CHECK
                # ----------------------------------------------------

                if x_column == y_column:

                    st.warning(
                        f"⚠️ X axis and Y axis are both "
                        f"'{x_column}'. Please select two "
                        f"different columns for a scatter plot."
                    )

                    st.info(
                        "A scatter plot needs two different "
                        "variables to show a relationship."
                    )

                else:

                    scatter_df = df[
                        [
                            x_column,
                            y_column
                        ]
                    ].dropna()

                    if scatter_df.empty:

                        st.warning(
                            "There is no usable data for these columns."
                        )

                    else:

                        st.markdown(
                            "#### 🔵 Visualization"
                        )

                        st.scatter_chart(
                            scatter_df,
                            x=x_column,
                            y=y_column,
                            use_container_width=True
                        )

# ============================================================
# CHAT TAB
# ============================================================

with chat_tab:

    # --------------------------------------------------------
    # LOAD / CREATE AGENT
    # --------------------------------------------------------

    agent = get_agent(
        chat_id,
        df
    )


    # --------------------------------------------------------
    # CONVERSATION HISTORY
    # --------------------------------------------------------

    for message in agent["messages"]:

        # Skip anything that isn't a normal dictionary
        if not isinstance(
            message,
            dict
        ):
            continue


        role = message.get(
            "role"
        )


        # ----------------------------------------------------
        # SYSTEM MESSAGE
        # ----------------------------------------------------

        if role == "system":

            continue


        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        if role == "user":

            with st.chat_message(
                "user"
            ):

                st.write(
                    message.get(
                        "content",
                        ""
                    )
                )


        # ----------------------------------------------------
        # ASSISTANT MESSAGE
        # ----------------------------------------------------

        elif role == "assistant":

            content = message.get(
                "content",
                ""
            )


            if content:

                with st.chat_message(
                    "assistant"
                ):

                    st.write(
                        content
                    )


    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask a question about your dataset..."
    )


    # --------------------------------------------------------
    # PROCESS QUESTION
    # --------------------------------------------------------

    if question:

        # ================================================
        # USER MESSAGE
        # ================================================

        with st.chat_message(
            "user"
        ):

            st.write(
                question
            )


        # Save user message to persistent storage

        storage.save_message(
            chat_id,
            "user",
            question
        )


        # ================================================
        # ASSISTANT
        # ================================================

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Analyzing your data..."
            ):

                response = run_agent(
                    agent,
                    question
                )


            # ============================================
            # TEXT RESPONSE
            # ============================================

            st.write(
                response
            )


            # ============================================
            # AI VISUALIZATION
            # ============================================

            with st.spinner(
                "Checking whether a visualization would help..."
            ):

                visualization_plan = (
                    get_visualization_plan(
                        question,
                        df
                    )
                )


            render_ai_visualization(
                visualization_plan,
                df
            )


        # ================================================
        # SAVE ASSISTANT RESPONSE
        # ================================================

        storage.save_message(
            chat_id,
            "assistant",
            response
        )