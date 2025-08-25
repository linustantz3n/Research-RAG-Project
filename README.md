# Research RAG Project

A comprehensive implementation and evaluation of Retrieval-Augmented Generation (RAG) using ChromaDB as the vector database. This project demonstrates document ingestion, semantic search, and performance benchmarking for question-answering tasks.

## Features

- **Document Processing**: Automated ingestion of PDF and Markdown documents
- **Vector Database**: ChromaDB implementation for efficient semantic search
- **RAG Pipeline**: Complete question-answering system with context retrieval
- **Performance Evaluation**: Comprehensive metrics and benchmarking suite
- **Test Suite**: Curated questions for systematic evaluation

## Tech Stack

- **Python 3.x**
- **ChromaDB** - Vector database for semantic search
- **LangChain** - Framework for LLM applications
- **OpenAI API** - Embeddings and language model
- **PyPDF** - PDF document processing
- **Unstructured** - Document parsing and chunking

## Project Structure

```
research_rag_proj/
├── buildDatabase.py      # Database construction and document ingestion
├── query.py             # Main query interface for RAG system
├── evaluation_metrics.py # Performance evaluation utilities
├── test_rag_performance.py # Comprehensive test suite
├── test_questions.py    # Curated test questions
├── data/               # Document storage
│   └── NIPS-2017-attention-is-all-you-need-Paper.pdf
├── chroma/             # ChromaDB vector database (auto-generated)
├── requirements.txt    # Python dependencies
└── .env               # Environment variables (create this)
```

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/linustantz3n/Research-RAG-Project.git
   cd Research-RAG-Project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### 1. Build the Database

First, process and index your documents:

```bash
python buildDatabase.py
```

This script will:
- Load PDF and Markdown files from the `data/` directory
- Split documents into chunks (1200 chars with 500 overlap)
- Generate embeddings using OpenAI
- Store vectors in ChromaDB

### 2. Query the System

Interactive question-answering:

```bash
python query.py
```

The system will:
- Accept your question via input prompt
- Perform semantic search to find relevant chunks
- Generate context-aware responses using RAG
- Display both the answer and source context

### 3. Run Performance Evaluation

Evaluate system performance with test questions:

```bash
python test_rag_performance.py
```

This will run the full evaluation suite and generate detailed metrics.

## How It Works

### Document Processing
1. **Loading**: PDFs are parsed page-by-page and combined to preserve context
2. **Chunking**: Documents are split using recursive character splitting
3. **Embedding**: Text chunks are converted to vectors using OpenAI embeddings
4. **Storage**: Vectors are stored in ChromaDB for efficient retrieval

### Query Processing
1. **Query Embedding**: User question is converted to vector representation
2. **Similarity Search**: Top-k most similar chunks are retrieved (k=4)
3. **Context Assembly**: Retrieved chunks are combined as context
4. **Response Generation**: LLM generates answer based on retrieved context
5. **Relevance Filtering**: Results below 0.65 similarity threshold are filtered out

## Dataset

The project includes the influential "Attention Is All You Need" paper (Vaswani et al., NIPS 2017) as a test document. This paper introduces the Transformer architecture and provides rich technical content for RAG evaluation.

## Performance Metrics

The evaluation suite includes:
- **Relevance Scoring**: Semantic similarity between queries and retrieved context
- **Response Quality**: Answer accuracy and completeness
- **Retrieval Precision**: Effectiveness of document chunk retrieval
- **End-to-End Performance**: Complete RAG pipeline evaluation

## Configuration

Key parameters in the system:
- **Chunk Size**: 1200 characters
- **Chunk Overlap**: 500 characters  
- **Retrieval Count**: Top 4 most similar chunks
- **Similarity Threshold**: 0.65 minimum relevance score
- **Embedding Model**: OpenAI text-embedding-ada-002

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Research Context

This implementation serves as a foundation for research into:
- RAG system optimization
- Vector database performance comparison
- Chunk size and overlap strategies
- Embedding model effectiveness
- LLM response quality measurement

## Acknowledgments

- Built with [LangChain](https://langchain.com/) framework
- Powered by [ChromaDB](https://www.trychroma.com/) vector database
- Uses [OpenAI](https://openai.com/) embeddings and language models