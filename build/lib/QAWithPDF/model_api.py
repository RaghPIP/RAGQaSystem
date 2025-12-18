import os
import sys
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from QAWithPDF.execption import customexception
from QAWithPDF.logger import setup_logger

load_dotenv()
logger = setup_logger()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_model():
    """
    Load OpenAI LLM.
    """
    try:
        logger.info("Loading OpenAI model")

        llm = OpenAI(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY
        )

        return llm

    except Exception as e:
        logger.error("Error loading OpenAI model", exc_info=True)
        raise customexception(e, sys)
