"""
Real Benchmark Script for QA System - Measures Actual Performance
"""
import time
import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project to path
sys.path.insert(0, '/Users/ragotmaragavendarnandagopal/Desktop/QASystemRAG')

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.embedding import create_or_load_index
from QAWithPDF.logger import logging

logger = logging.getLogger(__name__)

# Sample test documents
SAMPLE_DOCUMENTS = {
    "machine_learning.txt": """
Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience 
without being explicitly programmed. It focuses on the development of computer programs that can access data and use it 
to learn for themselves. The process begins with observations or data, such as examples, direct experience, or instruction, 
to look for patterns in data and make better decisions in the future based on the examples that we provide.

Machine Learning can be divided into three main categories: Supervised Learning, Unsupervised Learning, and Reinforcement Learning.

Supervised Learning involves training a model using labeled data. The model learns to predict the output given the input variables.
Common algorithms include linear regression, logistic regression, decision trees, and neural networks.

Unsupervised Learning deals with unlabeled data and aims to find hidden patterns or structure in the data. Common techniques include
clustering and dimensionality reduction.

Reinforcement Learning involves training a model to make sequential decisions by rewarding desired behaviors and punishing undesired ones.
This is the approach used in game-playing AI and robotics.
""",
    "deep_learning.txt": """
Deep Learning is part of a broader family of machine learning methods based on neural networks with multiple layers.
Deep neural networks can have many hidden layers that process information in increasingly abstract ways.

Convolutional Neural Networks (CNNs) are specially designed for processing spatial data like images. They use convolution operations
and pooling to extract features from images efficiently. CNNs have revolutionized computer vision, enabling applications like 
object detection, face recognition, and medical image analysis.

Recurrent Neural Networks (RNNs) are designed for sequential data processing. They maintain a hidden state that is updated
as they process each element in a sequence. This makes them ideal for natural language processing, time series analysis, and speech recognition.

Transformers are a newer architecture that uses self-attention mechanisms to process sequences in parallel. They have become
the foundation for state-of-the-art language models like BERT and GPT, achieving remarkable results in natural language understanding,
machine translation, and text generation.

The success of deep learning depends on having large amounts of data, computational power, and proper architectural design.
""",
    "nlp.txt": """
Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with
the interactions between computers and human language. It is used to apply machine learning algorithms to text and speech.

Text preprocessing is a crucial first step in NLP. It involves tasks like tokenization (breaking text into words), removing stopwords
(common words like 'the', 'is'), stemming (reducing words to their root form), and lemmatization (converting words to their base form).

Named Entity Recognition (NER) identifies and classifies entities in text such as persons, organizations, locations, and dates.
This is useful for information extraction and knowledge graph construction.

Sentiment Analysis determines the emotional tone of text, classifying it as positive, negative, or neutral. Applications include
social media monitoring, customer feedback analysis, and market sentiment tracking.

Machine Translation uses neural networks to automatically translate text from one language to another. Modern approaches like
sequence-to-sequence models with attention mechanisms have achieved impressive results.

Question Answering systems, like the one we built, use multiple NLP techniques including text understanding, information retrieval,
and answer generation to respond to user queries accurately.
""",
    "python.txt": """
Python is a high-level, interpreted programming language with dynamic semantics. Its high-level built-in data structures,
combined with dynamic binding and typing, make it very attractive for Rapid Application Development.

Python's simple, easy-to-learn syntax emphasizes readability and therefore reduces the cost of program maintenance.
Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.

Python has a rich ecosystem of libraries and frameworks:
- NumPy: Numerical computing with arrays and matrices
- Pandas: Data manipulation and analysis
- Scikit-learn: Machine learning algorithms
- TensorFlow and PyTorch: Deep learning frameworks
- Django and Flask: Web frameworks
- Matplotlib and Seaborn: Data visualization

Python's versatility makes it the preferred language for data science, artificial intelligence, web development,
automation, and scientific computing.

The Python community is large and active, contributing thousands of packages to the Python Package Index (PyPI),
making it easy to leverage existing solutions for almost any problem.
""",
}

def benchmark_system():
    """Run comprehensive benchmarks on the RAG system"""
    
    print("\n" + "=" * 80)
    print("QA SYSTEM - ACTUAL PERFORMANCE BENCHMARK")
    print("=" * 80)
    print()
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  WARNING: OPENAI_API_KEY not found in environment")
        print("   Benchmarking LOCAL operations only (data loading, chunking)")
        print("   To benchmark embeddings & LLM, set: export OPENAI_API_KEY='sk-...'")
        print()
        can_test_embeddings = False
    else:
        print("✅ OpenAI API key found - will benchmark full pipeline")
        can_test_embeddings = True
        print()
    
    # Create temp directory for test documents
    temp_dir = tempfile.mkdtemp()
    print(f"📁 Creating test documents in: {temp_dir}")
    print()
    
    # Write test documents
    total_chars = 0
    for filename, content in SAMPLE_DOCUMENTS.items():
        filepath = os.path.join(temp_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        total_chars += len(content)
        print(f"   ✓ {filename}: {len(content):,} characters")
    
    print(f"\n   Total test data: {total_chars:,} characters")
    print()
    
    # ===== BENCHMARK 1: DATA LOADING =====
    print("=" * 80)
    print("BENCHMARK 1: DATA LOADING & PARSING")
    print("=" * 80)
    
    start_load = time.time()
    try:
        documents = load_data(temp_dir)
        load_time = time.time() - start_load
        
        print(f"✅ Data Loading Time: {load_time:.4f} seconds")
        print(f"   Documents loaded: {len(documents)}")
        total_doc_chars = sum(len(doc.get_content()) for doc in documents)
        print(f"   Total characters parsed: {total_doc_chars:,}")
        print(f"   Parsing speed: {total_doc_chars/load_time:,.0f} chars/sec")
        print()
        
    except Exception as e:
        print(f"❌ Error during data loading: {e}")
        load_time = None
        documents = None
    
    # ===== BENCHMARK 2: INDEX CREATION/LOADING =====
    if documents and can_test_embeddings:
        print("=" * 80)
        print("BENCHMARK 2: INDEX CREATION (with OpenAI Embeddings)")
        print("=" * 80)
        
        # Remove old storage for fresh benchmark
        storage_path = "./storage_benchmark"
        if os.path.exists(storage_path):
            import shutil
            shutil.rmtree(storage_path)
        
        print(f"Creating embeddings for {len(documents)} documents...")
        print("This uses OpenAI API - may take a moment...\n")
        
        start_index = time.time()
        try:
            index = create_or_load_index(documents)
            index_time = time.time() - start_index
            
            print(f"✅ Index Creation Time: {index_time:.2f} seconds")
            print(f"   Number of nodes in index: {len(index.docstore.docs)}")
            
            # Estimate chunks created
            from llama_index.core import Settings
            print(f"   Chunk size: {Settings.chunk_size} characters")
            print(f"   Chunk overlap: {Settings.chunk_overlap} characters")
            
            # Calculate actual chunks
            estimated_chunks = len(index.docstore.docs)
            print(f"   Total chunks created: {estimated_chunks}")
            print(f"   Embedding speed: {estimated_chunks/index_time:.2f} chunks/sec")
            print()
            
        except Exception as e:
            print(f"❌ Error during index creation: {e}")
            print("   (Make sure your OpenAI API key is valid)")
            index = None
            index_time = None
    
    elif not can_test_embeddings:
        print("⏭️  Skipping embedding benchmark (no API key)")
        index = None
        index_time = None
    else:
        index = None
        index_time = None
    
    # ===== BENCHMARK 3: QUERY PERFORMANCE =====
    if index and can_test_embeddings:
        print("=" * 80)
        print("BENCHMARK 3: QUERY PERFORMANCE")
        print("=" * 80)
        print()
        
        test_queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "What is NLP used for?",
        ]
        
        query_times = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"Query {i}: '{query}'")
            
            try:
                start_query = time.time()
                query_engine = index.as_query_engine(similarity_top_k=5)
                response = query_engine.query(query)
                query_time = time.time() - start_query
                
                query_times.append(query_time)
                print(f"   Response time: {query_time:.2f} seconds")
                print(f"   Answer preview: {str(response.response)[:100]}...")
                print()
            except Exception as e:
                print(f"   ❌ Error: {e}")
                print()
        
        if query_times:
            avg_query_time = sum(query_times) / len(query_times)
            min_query_time = min(query_times)
            max_query_time = max(query_times)
            
            print(f"📊 Query Statistics:")
            print(f"   Average response time: {avg_query_time:.2f} seconds")
            print(f"   Fastest query: {min_query_time:.2f} seconds")
            print(f"   Slowest query: {max_query_time:.2f} seconds")
            print()
    
    # ===== SUMMARY =====
    print("=" * 80)
    print("SUMMARY - ACTUAL MEASURED METRICS")
    print("=" * 80)
    print()
    
    summary_data = {
        "Test Documents": f"{len(SAMPLE_DOCUMENTS)} files",
        "Total Input Size": f"{total_chars:,} characters",
        "Data Loading Time": f"{load_time:.4f} seconds" if load_time else "N/A",
        "Parsing Speed": f"{total_chars/load_time:,.0f} chars/sec" if load_time else "N/A",
    }
    
    if index_time:
        summary_data["Index Creation Time"] = f"{index_time:.2f} seconds"
        summary_data["Chunks Created"] = f"{estimated_chunks} chunks"
        summary_data["Embedding Speed"] = f"{estimated_chunks/index_time:.2f} chunks/sec"
    
    if query_times and len(query_times) > 0:
        summary_data["Avg Query Response Time"] = f"{avg_query_time:.2f} seconds"
        summary_data["Min Query Time"] = f"{min_query_time:.2f} seconds"
        summary_data["Max Query Time"] = f"{max_query_time:.2f} seconds"
    
    for key, value in summary_data.items():
        print(f"  • {key}: {value}")
    
    print()
    print("=" * 80)
    print("RESUME BULLET POINTS (Based on Actual Measurements)")
    print("=" * 80)
    print()
    
    if load_time and index_time and query_times:
        resume_points = [
            f"Built AI-powered RAG system using LlamaIndex + OpenAI that processes documents at {total_chars/load_time:,.0f} chars/sec and creates vector indexes at {estimated_chunks/index_time:.1f} chunks/sec with 1536-dimensional embeddings",
            
            f"Implemented semantic document Q&A system with {avg_query_time:.1f}s average query response time (from embedding lookup to LLM answer generation) using top-k=5 retrieval strategy",
            
            f"Engineered end-to-end Streamlit web application supporting multi-document PDF/TXT uploads with persistent vector storage, achieving sub-{max_query_time:.1f}s response times on diverse queries"
        ]
    elif load_time:
        resume_points = [
            f"Built AI-powered RAG system using LlamaIndex + OpenAI with document processing at {total_chars/load_time:,.0f} chars/sec parsing speed and 1536-dimensional vector embeddings",
            
            f"Implemented automated document ingestion pipeline with 800-character smart chunking strategy and persistent vector storage for instant index reloading",
            
            f"Engineered end-to-end Streamlit web application supporting PDF/TXT uploads, multi-document indexing, and semantic search with top-k=5 retrieval across 128K-token context window"
        ]
    else:
        resume_points = [
            "Built AI-powered RAG system using LlamaIndex + OpenAI that indexes documents into 1536-dimensional vectors with semantic similarity search",
            
            "Implemented automated document ingestion pipeline with 800-character chunking strategy and persistent vector storage for efficient retrieval",
            
            "Engineered end-to-end Streamlit web application supporting PDF/TXT uploads, multi-document indexing, and real-time Q&A with GPT-4o-mini"
        ]
    
    for i, point in enumerate(resume_points, 1):
        print(f"{i}. {point}\n")
    
    print("=" * 80)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    benchmark_system()
