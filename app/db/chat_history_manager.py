from langchain_core.messages import AIMessage, HumanMessage

from config import PATH_TO_DATABASE

import sqlite3
import os

class ChatHistoryManager():

    def __init__(self):

        save_path = os.path.join(PATH_TO_DATABASE,"chat_history")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # --- Define Database Structure ---    
        self.conn = sqlite3.connect(os.path.join(save_path,"chat_memory.db"),check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.__create_table()

    def __create_table(self)->None:
        """Creates initial chat_history table"""
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ChatHistory (
                session_id INTEGER Primary Key,
                uid INTEGER NOT NULL,
                msg_role TEXT NOT NULL,
                content TEXT NOT NULL,

                FOREIGN KEY (uid) REFERENCES Users(uid)
                    ON DELETE CASCADE
            )
            """,
        )
        self.conn.commit()

    def save_messages(self,session_id:str,msg_role:str,content:str) -> None:
        """Saves session_id, msg_role, and content to the chat_history table"""
        self.cursor.execute(
            """
            INSERT INTO chat_history (session_id,msg_role,content)
            VALUES (?,?,?)
            """,
            (session_id,msg_role,content)
        )

        self.conn.commit()

    def load_message_history(self,session_id:str,k:int = 10):
        """Returns the first k number of role and content of a given session_id chat_history"""
        self.cursor.execute(
            """
            SELECT role,content FROM chat_history WHERE session_id = ?
            LIMIT ?
            """,
            (session_id,k)
        )

        chat_history = []
        rows = self.cursor.fetchall()

        for role,content in rows:
            # We will only append human and ai messages
            if role == "human":
                chat_history.append(HumanMessage(content=content))
            elif role == "ai":
                chat_history.append(AIMessage(content=content))

        return chat_history
    

    def get_all_chats(self):
        """Returns all session_ids from chat_history table"""
        self.cursor.execute(
            """
            SELECT DISTINCT session_id FROM chat_history ORDER BY rowid DESC
            """
        )
        return [row[0] for row in self.cursor.fetchall()]


        



