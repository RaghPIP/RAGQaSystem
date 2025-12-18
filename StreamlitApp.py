import streamlit as st

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.embedding import create_or_load_index   # OpenAI-based index
from QAWithPDF.model_api import load_model                 # OpenAI LLM

def main():
    st.set_page_config(page_title="QA with Documents", layout="centered")

    st.header("📄 QA with Documents (OpenAI RAG)")

    uploaded_file = st.file_uploader(
        "Upload your document (PDF / TXT)",
        type=["pdf", "txt"]
    )

    user_question = st.text_input("Ask your question")

    if st.button("Submit & Process"):
        if uploaded_file is None:
            st.warning("Please upload a document first.")
            return

        if not user_question:
            st.warning("Please enter a question.")
            return

        with st.spinner("Processing document..."):
            # 1️⃣ Save uploaded file temporarily
            with open(f"Data/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 2️⃣ Load documents
            documents = load_data("Data")

            # 3️⃣ Create or load OpenAI-based index
            index = create_or_load_index(documents)

            # 4️⃣ Query
            query_engine = index.as_query_engine(similarity_top_k=5)
            response = query_engine.query(user_question)

            st.success("Answer:")
            st.write(response.response)

if __name__ == "__main__":
    main()
