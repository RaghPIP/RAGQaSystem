import os
import sys

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.execption import customexception
from QAWithPDF.logger import setup_logger

logger = setup_logger()

PERSIST_DIR = "./storage"

def create_or_load_index(documents):
    """
    Create or load a vector index using OpenAI embeddings.
    """
    try:
        logger.info("Initializing OpenAI models")

        Settings.llm = OpenAI(model="gpt-4o-mini")
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small"
        )
        Settings.chunk_size = 800
        Settings.chunk_overlap = 20

        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            logger.info("Loading existing index from storage")
            storage_context = StorageContext.from_defaults(
                persist_dir=PERSIST_DIR
            )
            index = load_index_from_storage(storage_context)
        else:
            logger.info("Creating new index")
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=PERSIST_DIR)

        return index

    except Exception as e:
        logger.error("Error creating/loading index", exc_info=True)
        raise customexception(e, sys)
