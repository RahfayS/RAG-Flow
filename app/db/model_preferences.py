from config import PATH_TO_DATABASE

import streamlit as st
import sqlite3
import os


class ModelPreferencesDB():

    def __init__(self):

        save_path = os.path.join(PATH_TO_DATABASE,"model_preferences")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # --- Define Database Structure ---   
        self.db_path = os.path.join(save_path,"preferences.db")
        self.__create_table()


    def __create_table(self)->None:
        """Creates initial user table"""
        with sqlite3.connect(self.db_path,timeout=10) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ModelPreferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uid INTEGER NOT NULL,

                    save_name NOT NULL UNIQUE,
                    temperature REAL NOT NULL CHECK (temperature <= 1.0),
                    max_tokens INTEGER NOT NULL CHECK (max_tokens <= 4096),
                    model TEXT NOT NULL,
                    top_k_chunks INTEGER NOT NULL CHECK (top_k_chunks <= 25),
                    chunk_size INTEGER NOT NULL CHECK (chunk_size <= 1048),
                    chunk_overlap INTEGER NOT NULL CHECK (chunk_overlap <= 216),
                    score_threshold REAL NOT NULL CHECK (score_threshold <= 1.0),

                    chunk_method TEXT NOT NULL CHECK (
                        chunk_method IN (
                            'RecursiveCharacterTextSplitter',
                            'SentenceTransformerTokenTextSplitter'
                        )
                    ),

                    top_n INTEGER,

                    FOREIGN KEY (uid) REFERENCES Users(uid)
                        ON DELETE CASCADE
                )
                """
            )
            conn.commit()
            return True

    def __validate_insert(self,cursor,uid:int,k:int = 5):
        """Validates user does not have too many preferences"""
        cursor.execute(
            """
            SELECT COUNT(*) FROM ModelPreferences
            WHERE uid = ?
                """,
                (uid,)
        )
        count = cursor.fetchone()[0]

        if count >= k: 
            raise ValueError(f"User already has maximum number ({k}) of model preferences, please delete a preference before adding a new one")


    def add_preferences(self,uid:int,save_name:str,temp:float,max_tokens:int,model:str,top_k:int,chunk_size:int,chunk_overlap:int,score_thresh:float,chunk_method:str,top_n:int) -> bool:
        """Adds user config preferences to ModelPreferences"""
        with sqlite3.connect(self.db_path,timeout=10) as conn:
            cursor = conn.cursor() 
            try:
                self.__validate_insert(cursor,uid)

                cursor.execute(
                    """
                    INSERT INTO ModelPreferences (uid,save_name,temperature,max_tokens,model,top_k_chunks,chunk_size,chunk_overlap,score_threshold,chunk_method,top_n)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (uid,save_name,temp,max_tokens,model,top_k,chunk_size,chunk_overlap,score_thresh,chunk_method,top_n,)

                )
                conn.commit()
                return True
            except sqlite3.IntegrityError as e:
                return False
            except ValueError as e:
                st.error(e)
                return False


    def load_preferences(self,uid:int):
        """Returns preference from ModelPreferences table using uid"""
        with sqlite3.connect(self.db_path,timeout=10) as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM ModelPreferences WHERE ModelPreferences.uid = ?
                    """,
                    (uid,)
                )

                return cursor.fetchall()
            except sqlite3.Error as e:
                return []

    
    def delete_preferences(self,id):
        """Deletes preference from ModelPreferences table"""
        with sqlite3.connect(self.db_path,timeout=10) as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    DELETE FROM ModelPreferences WHERE ModelPreferences.id = ?
                    """,
                    (id,)
                )

                conn.commit()
                return True
                


            except sqlite3.Error as e:
                return False