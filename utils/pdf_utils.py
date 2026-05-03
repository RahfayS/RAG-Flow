from config import SESSIONS_DIR

from typing import Any
import streamlit as st
import os

def save_uploaded_files(files: list[Any], session_id: str)->None:
    """Saves user uploaded files to disk"""

    if not files:
        return

    user_dir = os.path.join(SESSIONS_DIR,f"session_{session_id}/documents")
    os.makedirs(user_dir, exist_ok=True)

    with st.expander("Saved Status"):
        for file in files:
            if file is None:
                st.warning(f'file cannot be found, skipping')
                continue

            file_path = os.path.join(user_dir, file.name)

            if os.path.exists(file_path):
                st.warning(f"{file.name} already exists")
                continue
            
            st.success(f"Saving: {file.name} ({file.size} bytes)")

            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            