from langchain_core.messages import AIMessage, HumanMessage

from config import PATH_TO_CHAT_MEMORY

import sqlite3
import os

class ChatHistoryManager():

    def __init__(self):

        if not os.path.exists(PATH_TO_CHAT_MEMORY):
            os.makedirs(PATH_TO_CHAT_MEMORY)

        # --- Define Database Structure ---    
        self.conn = sqlite3.connect(os.path.join(PATH_TO_CHAT_MEMORY,"chat_memory.db"),check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.__create_table()

    def __create_table(self)->None:
        """Creates initial chat_history table"""
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                chat_id TEXT,
                msg_role TEXT,
                content TEXT
            )
            """,
        )
        self.conn.commit()

    def save_messages(self,chat_id:str,msg_role:str,content:str) -> None:
        """Saves chat_id, msg_role, and content to the chat_history table"""
        self.cursor.execute(
            """
            INSERT INTO chat_history (chat_id,msg_role,content)
            VALUES (?,?,?)
            """,
            (chat_id,msg_role,content)
        )

        self.conn.commit()

    def load_message_history(self,chat_id:str,k:int = 10):
        """Returns the first k number of role and content of a given chat_id chat_history"""
        self.cursor.execute(
            """
            SELECT role,content FROM chat_history WHERE chat_id = ?
            LIMIT ?
            """,
            (chat_id,k)
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
        """Returns all chat_ids from chat_history table"""
        self.cursor.execute(
            """
            SELECT DISTINCT chat_id FROM chat_history ORDER BY rowid DESC
            """
        )
        return [row[0] for row in self.cursor.fetchall()]


        



