import os

ROOT = os.getcwd()
SESSIONS_DIR = os.path.join(ROOT,"sessions")

DATA_DIR = os.path.join(ROOT,"data")
SESSIONS_DIR = os.path.join(DATA_DIR,"sessions")

VECTOR_DB_DIR = os.path.join(DATA_DIR,"vector_db")

PATH_TO_CHAT_MEMORY = os.path.join(ROOT,"memory")
LOGS_DIR = os.path.join(ROOT,"logs")
