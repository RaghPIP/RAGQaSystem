import logging
import os
import sys
import tempfile

from llama_index.core import SimpleDirectoryReader

try:
    # When imported as package
    from .execption import customexception
except ImportError:
    # When run as a script without package context
    from QAWithPDF.execption import customexception


def load_data(uploaded_file):
    """Load a single uploaded file (Streamlit) into documents for indexing."""
    if uploaded_file is None:
        raise customexception("No file uploaded", sys)

    tmp_path = None
    try:
        logging.info("Data loading started")

        if hasattr(uploaded_file, "read"):
            # Streamlit's UploadedFile; write to temp so SimpleDirectoryReader can consume it
            suffix = os.path.splitext(getattr(uploaded_file, "name", ""))[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            loader = SimpleDirectoryReader(input_files=[tmp_path])
        else:
            # Assume caller passed a path or directory
            loader = SimpleDirectoryReader(uploaded_file)

        documents = loader.load_data()
        logging.info("Data loading completed")
        return documents
    except Exception as e:
        logging.error("Exception in data ingestion", exc_info=True)
        raise customexception(e, sys)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
