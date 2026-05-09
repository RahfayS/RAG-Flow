from config import PATH_TO_DATABASE

import streamlit as st
import sqlite3
import os


class UserDB():

    def __init__(self):

        save_path = os.path.join(PATH_TO_DATABASE,"users")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # --- Define Database Structure ---   
        self.db_path = os.path.join(save_path,"users.db")
        self.__create_table()


    def __create_table(self)->None:
        """Creates initial user table"""
        with sqlite3.connect(self.db_path,timeout=10) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS Users (
                    uid INTEGER PRIMARY KEY,
                    sub TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL,
                    name TEXT NOT NULL
                )
                """
            )
            conn.commit()
            return True

    def add_user(self,sub:str,email:str,name:str):
        """Adds user to Users table using sub (unique ID google assigns a user), email, and name"""
        with sqlite3.connect(self.db_path,timeout=10) as conn:
            cursor = conn.cursor() 
            try:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO Users (sub,email,name)
                    VALUES (?,?,?)
                    """,
                    (sub,email,name)
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError as e:
                st.warning("User already exists")
                return False


    def load_user(self,uid:int):
        """Returns user row from Users table using uid"""
        with sqlite3.connect(self.db_path,timeout=10) as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM Users WHERE Users.uid = ?
                    """,
                    (uid)
                )

                return cursor.fetchall()
            except sqlite3.IntegrityError as e:
                return False

    
    def remove_user(self,uid:int):
        """Removes a user from Users"""
        with sqlite3.connect(self.db_path,timeout=10) as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    DELETE FROM Users WHERE Users.uid = ?
                    """,
                    (uid)
                )
                conn.commit()
                return True
            
            except sqlite3.IntegrityError as e:
                return False
            
    def get_uid(self,sub:str) -> int | None:
        with sqlite3.connect(self.db_path,timeout=10) as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT uid FROM Users WHERE Users.sub = ?
                    """,
                    (sub,)
                )
                return cursor.fetchone()[0]
            except sqlite3.IntegrityError as e:
                return None