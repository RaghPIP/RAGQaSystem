from llama_index.core import SimpleDirectoryReader
import sys
import logging

from execption import customexception

def load_data(data_path):
    try:
        logging.info("Data loading started")
        loader = SimpleDirectoryReader(data_path)
        documents = loader.load_data()
        logging.info("Data loading completed")
        return documents
    except Exception as e:
        logging.error("Exception in data ingestion", exc_info=True)
        raise customexception(e, sys)
