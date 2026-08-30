import os
import sqlite3
import uuid
from datetime import datetime


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "app.db"
)

UPLOADS_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)


os.makedirs(
    UPLOADS_DIR,
    exist_ok=True
)


# ============================================================
# DATABASE
# ============================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chats (

            id TEXT PRIMARY KEY,

            title TEXT NOT NULL,

            dataset_filename TEXT,

            dataset_path TEXT,

            created_at TEXT NOT NULL,

            updated_at TEXT NOT NULL
        )
        """
    )


    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            chat_id TEXT NOT NULL,

            role TEXT NOT NULL,

            content TEXT NOT NULL,

            created_at TEXT NOT NULL,

            FOREIGN KEY(chat_id)
                REFERENCES chats(id)
                ON DELETE CASCADE
        )
        """
    )


    connection.commit()

    connection.close()


# ============================================================
# CREATE CHAT
# ============================================================

def create_chat(
    title="New Chat"
):

    chat_id = str(
        uuid.uuid4()
    )

    now = datetime.utcnow().isoformat()

    connection = get_connection()

    connection.execute(
        """
        INSERT INTO chats (
            id,
            title,
            created_at,
            updated_at
        )

        VALUES (?, ?, ?, ?)
        """,
        (
            chat_id,
            title,
            now,
            now
        )
    )

    connection.commit()

    connection.close()

    return chat_id


# ============================================================
# GET ALL CHATS
# ============================================================

def get_chats():

    connection = get_connection()

    chats = connection.execute(
        """
        SELECT *
        FROM chats
        ORDER BY updated_at DESC
        """
    ).fetchall()

    connection.close()

    return chats


# ============================================================
# GET ONE CHAT
# ============================================================

def get_chat(
    chat_id
):

    connection = get_connection()

    chat = connection.execute(
        """
        SELECT *
        FROM chats
        WHERE id = ?
        """,
        (chat_id,)
    ).fetchone()

    connection.close()

    return chat


# ============================================================
# DELETE CHAT
# ============================================================

def delete_chat(chat_id):

    chat = get_chat(
        chat_id
    )

    if chat is None:

        return


    dataset_path = chat[
        "dataset_path"
    ]


    connection = get_connection()

    connection.execute(
        """
        DELETE FROM chats
        WHERE id = ?
        """,
        (
            chat_id,
        )
    )


    connection.commit()

    connection.close()


    # -----------------------------------------------
    # Delete physical dataset
    # -----------------------------------------------

    if (
        dataset_path
        and os.path.exists(dataset_path)
    ):

        try:

            os.remove(
                dataset_path
            )

        except OSError:

            pass

# ============================================================
# UPDATE CHAT
# ============================================================

def update_chat(
    chat_id,
    title=None,
    dataset_filename=None,
    dataset_path=None
):

    connection = get_connection()

    current = get_chat(
        chat_id
    )

    if current is None:

        connection.close()

        return


    new_title = (
        title
        if title is not None
        else current["title"]
    )

    new_filename = (
        dataset_filename
        if dataset_filename is not None
        else current["dataset_filename"]
    )

    new_path = (
        dataset_path
        if dataset_path is not None
        else current["dataset_path"]
    )


    now = datetime.utcnow().isoformat()


    connection.execute(
        """
        UPDATE chats

        SET
            title = ?,
            dataset_filename = ?,
            dataset_path = ?,
            updated_at = ?

        WHERE id = ?
        """,
        (
            new_title,
            new_filename,
            new_path,
            now,
            chat_id
        )
    )


    connection.commit()

    connection.close()


# ============================================================
# SAVE MESSAGE
# ============================================================

def save_message(
    chat_id,
    role,
    content
):

    now = datetime.utcnow().isoformat()

    connection = get_connection()

    connection.execute(
        """
        INSERT INTO messages (
            chat_id,
            role,
            content,
            created_at
        )

        VALUES (?, ?, ?, ?)
        """,
        (
            chat_id,
            role,
            content,
            now
        )
    )


    connection.execute(
        """
        UPDATE chats

        SET updated_at = ?

        WHERE id = ?
        """,
        (
            now,
            chat_id
        )
    )


    connection.commit()

    connection.close()


# ============================================================
# GET CHAT MESSAGES
# ============================================================

def get_messages(
    chat_id
):

    connection = get_connection()

    messages = connection.execute(
        """
        SELECT role, content
        FROM messages

        WHERE chat_id = ?

        ORDER BY id ASC
        """,
        (
            chat_id,
        )
    ).fetchall()

    connection.close()

    return messages


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(
    chat_id,
    uploaded_file
):

    # -----------------------------------------------
    # Get existing dataset
    # -----------------------------------------------

    chat = get_chat(
        chat_id
    )


    old_path = None

    if chat:

        old_path = chat[
            "dataset_path"
        ]


    # -----------------------------------------------
    # File extension
    # -----------------------------------------------

    extension = os.path.splitext(
        uploaded_file.name
    )[1].lower()


    # -----------------------------------------------
    # Chat-specific filename
    # -----------------------------------------------

    filename = (
        f"{chat_id}"
        f"{extension}"
    )


    path = os.path.join(
        UPLOADS_DIR,
        filename
    )


    # -----------------------------------------------
    # Save new dataset
    # -----------------------------------------------

    with open(
        path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )


    # -----------------------------------------------
    # Delete previous dataset
    # -----------------------------------------------

    if (
        old_path
        and old_path != path
        and os.path.exists(old_path)
    ):

        try:

            os.remove(
                old_path
            )

        except OSError:

            pass


    # -----------------------------------------------
    # Update database
    # -----------------------------------------------

    update_chat(
        chat_id,
        dataset_filename=uploaded_file.name,
        dataset_path=path
    )


    return path

initialize_database()

def rename_chat(chat_id, title):

    connection = get_connection()

    now = datetime.utcnow().isoformat()

    connection.execute(
        """
        UPDATE chats
        SET title = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            title,
            now,
            chat_id
        )
    )

    connection.commit()

    connection.close()

def chat_has_dataset(chat_id):

    chat = get_chat(chat_id)

    if chat is None:
        return False

    return chat["dataset_path"] is not None


def chat_has_messages(chat_id):

    messages = get_messages(chat_id)

    return len(messages) > 0


def is_empty_chat(chat_id):

    return (
        not chat_has_dataset(chat_id)
        and not chat_has_messages(chat_id)
    )