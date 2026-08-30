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

import storage


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


    elif extension == ".xlsx":

        return pd.read_excel(
            file_path,
            engine="openpyxl"
        )


    elif extension == ".xls":

        return pd.read_excel(
            file_path,
            engine="xlrd"
        )


    else:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# INITIALIZE STORAGE
# ============================================================

storage.initialize_database()


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv(
    "GROQ_API_KEY"
)


# ============================================================
# GROQ CLIENT
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


# ============================================================
# HELPER: CREATE AGENT FOR CHAT
# ============================================================

def get_agent(
    chat_id,
    df
):

    if chat_id not in st.session_state.agents:

        data_agent = create_data_agent(
            df
        )

        TOOLS = create_tools(
            data_agent
        )

        agent = create_agent(
            client,
            TOOLS
        )

        # Load previous messages
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
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("💬 Conversations")

    st.divider()


    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    chats = storage.get_chats()

    empty_chat_id = None

    for chat in chats:

        if storage.is_empty_chat(chat["id"]):

            empty_chat_id = chat["id"]
            break


    if st.button(
        "＋ New Chat",
        use_container_width=True
    ):

        # Do not allow multiple empty chats

        if empty_chat_id:

            st.session_state.active_chat_id = (
                empty_chat_id
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


    # --------------------------------------------------------
    # CHAT LIST
    # --------------------------------------------------------

    chats = storage.get_chats()


    if not chats:

        st.caption(
            "No conversations yet."
        )


    for chat in chats:

        chat_id = chat["id"]

        title = chat["title"]


        # -----------------------------------------------
        # Chat row
        # -----------------------------------------------

        chat_col, menu_col = st.columns(
            [5, 1],
            gap="small"
        )


        with chat_col:

            if st.button(
                f"💬 {title}",
                key=f"chat_{chat_id}",
                use_container_width=True
            ):

                current_chat_id = (
                    st.session_state.active_chat_id
                )


                # Delete abandoned empty chat

                if (
                    current_chat_id
                    and current_chat_id != chat_id
                    and storage.is_empty_chat(
                        current_chat_id
                    )
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


                st.session_state.active_chat_id = (
                    chat_id
                )

                st.rerun()


        # -----------------------------------------------
        # Three-dot menu
        # -----------------------------------------------

        with menu_col:

            with st.popover(
                "⋮",
                use_container_width=True
            ):

                st.caption(
                    title
                )


                # ----------------------------
                # RENAME
                # ----------------------------

                if st.button(
                    "✏️ Rename",
                    key=f"rename_{chat_id}",
                    use_container_width=True
                ):

                    st.session_state[
                        "renaming_chat_id"
                    ] = chat_id

                    st.rerun()


                # ----------------------------
                # DELETE
                # ----------------------------

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


                    remaining_chats = (
                        storage.get_chats()
                    )


                    if remaining_chats:

                        st.session_state.active_chat_id = (
                            remaining_chats[0]["id"]
                        )

                    else:

                        st.session_state.active_chat_id = (
                            None
                        )


                    st.rerun()


        # -----------------------------------------------
        # Inline rename interface
        # -----------------------------------------------

        if (
            st.session_state.get(
                "renaming_chat_id"
            )
            == chat_id
        ):

            new_title = st.text_input(
                "Chat name",
                value=title,
                key=f"rename_input_{chat_id}"
            )


            rename_col1, rename_col2 = st.columns(
                2
            )


            with rename_col1:

                if st.button(
                    "Save",
                    key=f"save_rename_{chat_id}",
                    use_container_width=True
                ):

                    cleaned_title = (
                        new_title.strip()
                    )


                    if cleaned_title:

                        storage.rename_chat(
                            chat_id,
                            cleaned_title
                        )


                    st.session_state[
                        "renaming_chat_id"
                    ] = None

                    st.rerun()


            with rename_col2:

                if st.button(
                    "Cancel",
                    key=f"cancel_rename_{chat_id}",
                    use_container_width=True
                ):

                    st.session_state[
                        "renaming_chat_id"
                    ] = None

                    st.rerun()


    st.divider()


# ============================================================
# MAIN AREA
# ============================================================

st.title("📊 AI Data Analyst")


# ============================================================
# NO ACTIVE CHAT
# ============================================================

if st.session_state.active_chat_id is None:

    st.info(
        "Create a new chat or upload a dataset to begin."
    )


    uploaded_file = st.file_uploader(
        "Upload your dataset",
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

            st.error(
                f"Could not read dataset: {e}"
            )

            st.stop()


        data_agent = create_data_agent(
            df
        )

        TOOLS = create_tools(
            data_agent
        )

        st.session_state.agents[
            chat_id
        ] = create_agent(
            client,
            TOOLS
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
# DATASET
# ============================================================

dataset_path = chat["dataset_path"]


if dataset_path is None:

    st.warning(
        "This chat does not have a dataset yet."
    )


    uploaded_file = st.file_uploader(
        "Upload a dataset for this chat",
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
# CHAT HEADER
# ============================================================

header_col, action_col = st.columns(
    [6, 2]
)


with header_col:

    st.subheader(
        chat["title"]
    )

    st.caption(
        f"📄 {chat['dataset_filename']} "
        f"• {len(df)} rows × {len(df.columns)} columns"
    )


with action_col:

    if st.button(
        "Rename Chat ✏️",
        key="header_rename_chat",
        use_container_width=True
    ):

        st.session_state[
            "renaming_chat_id"
        ] = chat_id

        st.rerun()


    if st.button(
        "Delete Chat 🗑️",
        key="header_delete_chat",
        use_container_width=True
    ):

        storage.delete_chat(
            chat_id
        )


        if chat_id in st.session_state.agents:

            del st.session_state.agents[
                chat_id
            ]


        remaining_chats = storage.get_chats()


        if remaining_chats:

            st.session_state.active_chat_id = (
                remaining_chats[0]["id"]
            )

        else:

            st.session_state.active_chat_id = None


        st.rerun()


# ------------------------------------------------------------
# Rename input
# ------------------------------------------------------------

if (
    st.session_state.get(
        "renaming_chat_id"
    )
    == chat_id
):

    rename_col1, rename_col2 = st.columns(
        [5, 1]
    )


    with rename_col1:

        new_title = st.text_input(
            "Chat name",
            value=chat["title"],
            key=f"header_rename_input_{chat_id}"
        )


    with rename_col2:

        if st.button(
            "Save",
            key=f"header_save_rename_{chat_id}",
            use_container_width=True
        ):

            cleaned_title = (
                new_title.strip()
            )


            if cleaned_title:

                storage.rename_chat(
                    chat_id,
                    cleaned_title
                )


            st.session_state[
                "renaming_chat_id"
            ] = None

            st.rerun()

# ============================================================
# DATASET PREVIEW
# ============================================================

with st.expander(
    "📋 Preview dataset"
):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


st.divider()


# ============================================================
# AGENT
# ============================================================

agent = get_agent(
    chat_id,
    df
)


# ============================================================
# CHAT HISTORY
# ============================================================

for message in agent["messages"]:

    if message["role"] == "system":

        continue


    if message["role"] == "user":

        with st.chat_message(
            "user"
        ):

            st.write(
                message["content"]
            )


    elif message["role"] == "assistant":

        with st.chat_message(
            "assistant"
        ):

            st.write(
                message["content"]
            )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your dataset..."
)


if question:

    with st.chat_message(
        "user"
    ):

        st.write(question)


    # Save user message
    storage.save_message(
        chat_id,
        "user",
        question
    )


    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Analyzing..."
        ):

            response = run_agent(
                agent,
                question
            )


        st.write(response)


    # Save assistant message
    storage.save_message(
        chat_id,
        "assistant",
        response
    )

    # --------------------------------------------------------
# RENAME CURRENT CHAT
# --------------------------------------------------------

if st.session_state.active_chat_id:

    current_chat = storage.get_chat(
        st.session_state.active_chat_id
    )
