# 🤖 AUGUS - Financial Agentic RAG System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-purple?style=for-the-badge)](https://langgraph.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen?style=for-the-badge)](tests/)

**Autonomous Financial Research Engine powered by AI Agents**

[Features](#✨-features) • [Quick Start](#🚀-quick-start) • [Architecture](#🏗️-architecture) • [API](#📡-api-endpoints) • [Contributing](#🤝-contributing)

</div>

---

## ✨ Features

<table>
  <tr>
    <td width="50%">
      <h3>🔍 Multi-Agent Analysis</h3>
      <ul>
        <li>Analyst Agent for deep financial insights</li>
        <li>Auditor Agent for compliance verification</li>
        <li>Retriever Agent for intelligent document sourcing</li>
      </ul>
    </td>
    <td width="50%">
      <h3>📊 Comprehensive Tools</h3>
      <ul>
        <li>SEC API integration for real-time filings</li>
        <li>Advanced financial calculations</li>
        <li>Vector-based document retrieval</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>🎯 RAG Pipeline</h3>
      <ul>
        <li>LangGraph state management</li>
        <li>Qdrant vector database</li>
        <li>Semantic document search</li>
      </ul>
    </td>
    <td width="50%">
      <h3>✅ Quality Assurance</h3>
      <ul>
        <li>RAGAS evaluation framework</li>
        <li>Human-in-the-loop approval</li>
        <li>Compliance reporting</li>
      </ul>
    </td>
  </tr>
</table>

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.13+**
- **Docker & Docker Compose** (optional, for containerized deployment)
- **Git** for version control

### Installation

#### Option 1: Local Setup

```bash
# Clone the repository
git clone https://github.com/Code-Master-1110-PGH/Augus.git
cd Augus

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access the application
# Web UI: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Running the Application

```bash
# Activate virtual environment (if not already activated)
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1

# Start the FastAPI server
python -m uvicorn app.main:app --reload

# Access the application
# 🌐 Web Interface: http://127.0.0.1:8000
# 📚 API Documentation: http://127.0.0.1:8000/docs
# 💚 Health Check: http://127.0.0.1:8000/health
```

---

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    AUGUS Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Retriever  │  │   Analyst    │  │   Auditor    │       │
│  │    Agent     │  │   Agent      │  │   Agent      │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                  │                │
│         └─────────────────┼──────────────────┘                │
│                           │                                    │
│                      ┌────▼─────┐                             │
│                      │   Graph   │                             │
│                      │  Manager  │                             │
│                      └────┬─────┘                             │
│                           │                                    │
│         ┌─────────────────┼─────────────────┐                │
│         │                 │                 │                │
│    ┌────▼────┐    ┌──────▼──────┐    ┌────▼────┐           │
│    │   SEC   │    │  Vector DB  │    │ Finance │           │
│    │   API   │    │   (Qdrant)  │    │ Calc    │           │
│    └────┬────┘    └─────────────┘    └────┬────┘           │
│         │                                  │                │
│         └──────────────┬───────────────────┘                │
│                        │                                    │
│                   ┌────▼────────┐                           │
│                   │   FastAPI   │                            │
│                   │  (REST API) │                            │
│                   └─────────────┘                            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Project Structure

```
Augus/
├── 📄 README.md                 # Project documentation
├── 📦 requirements.txt          # Python dependencies
├── 🐳 Dockerfile               # Docker configuration
├── 🔧 docker-compose.yml       # Docker Compose orchestration
│
├── app/                         # Main application directory
│   ├── __init__.py
│   ├── 🎛️ config.py            # Configuration management
│   ├── 🚀 main.py              # FastAPI application entry
│   │
│   ├── agents/                  # AI Agent implementations
│   │   ├── analyst_agent.py     # Financial analysis agent
│   │   ├── auditor_agent.py     # Compliance auditor agent
│   │   └── retriever_agent.py   # Document retriever agent
│   │
│   ├── core/                    # Core components
│   │   ├── graph.py             # LangGraph workflow definition
│   │   ├── providers.py         # LLM & tool providers
│   │   └── state.py             # State management schema
│   │
│   ├── tools/                   # Integrated tools
│   │   ├── sec_tools.py         # SEC API integration
│   │   ├── finance_calc.py      # Financial calculations
│   │   └── vector_store.py      # Vector database operations
│   │
│   ├── schemas/                 # Data models
│   │   └── models.py            # Pydantic models
│   │
│   ├── evaluation/              # Quality assurance
│   │   └── ragas_eval.py        # RAGAS evaluation framework
│   │
│   └── static/                  # Frontend assets
│       ├── index.html           # Main workspace UI
│       ├── landing.html         # Landing page
│       ├── login.html           # Authentication
│       ├── app.js               # Main application logic
│       └── styles.css           # Styling
│
└── tests/                       # Unit & integration tests
    ├── test_graph.py            # Graph workflow tests
    └── test_tools.py            # Tool integration tests
```

---

## 📡 API Endpoints

### Research Operations

#### Start Research
```http
POST /api/v1/research/start
Content-Type: application/json

{
  "query": "What are the financial risks in Apple?",
  "company_ticker": "AAPL"
}

Response (202 Accepted):
{
  "thread_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### Get Research Status
```http
GET /api/v1/research/{thread_id}/status

Response (200 OK):
{
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "question": "What are the financial risks in Apple?",
  "status": "AWAITING_HUMAN_REVIEW | APPROVED | RUNNING",
  "iteration_count": 3,
  "financial_analysis": { ... },
  "compliance_report": { ... },
  "retrieved_documents": 12,
  "graph_history": [ ... ]
}
```

#### Resume Research (Human Approval)
```http
POST /api/v1/research/{thread_id}/resume
Content-Type: application/json

{
  "approved": true,
  "reviewer": "analyst@company.com",
  "comment": "Analysis looks good, ready for publication"
}
```

### System Operations

#### Health Check
```http
GET /health

Response (200 OK):
{
  "status": "ok"
}
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# LLM Configuration
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_API_KEY=your_huggingface_token

# SEC API
SEC_API_KEY=your_sec_api_key

# Qdrant Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_key

# LangSmith (optional)
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=augus-project
```

### Local Development (No API Keys)

The application works out-of-the-box with DEMO ticker:

```bash
# Run with DEMO data (uses offline SEC filing fixture)
python -m uvicorn app.main:app --reload

# Query DEMO ticker (no keys required)
curl -X POST http://localhost:8000/api/v1/research/start \
  -H "Content-Type: application/json" \
  -d '{"query":"revenue growth","company_ticker":"DEMO"}'
```

---

## 🚀 Deployment

### Docker Deployment

```bash
# Build the Docker image
docker build -t augus:latest .

# Run with Docker
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key \
  -e SEC_API_KEY=your_key \
  augus:latest
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_graph.py -v

# Run tests with async support
pytest tests/ -v --asyncio-mode=auto

# Run evaluation
python -m app.evaluation.ragas_eval
```

---

## 📊 Performance Monitoring

### Key Metrics

- **Response Time**: Track API latency
- **Agent Decisions**: Monitor agent node execution
- **Document Retrieval**: Vector DB query performance
- **Human Review Rate**: Track approval bottlenecks

### LangSmith Integration

The project integrates with LangSmith for LLM observability:

```bash
# Enable in .env
LANGSMITH_API_KEY=your_token
LANGSMITH_TRACING=true
```

---

## 💡 Best Practices

### Agent Development

- Use deterministic calculations (don't rely on LLM for arithmetic)
- Call `get_reasoning_model()` or `get_fast_model()` for qualitative reasoning
- Implement proper error handling in agent nodes
- Always include compliance gates for sensitive operations

### Local Embeddings

- Uses Hugging Face embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- No OpenAI embedding model required
- Qdrant vector database for similarity search
- In-memory adapter by default (set `FINANCIAL_VECTOR_BACKEND=qdrant` for production)

### Production Deployment

- Replace `MemorySaver` with Postgres checkpointer
- Put API behind durable task queue
- Enable LangSmith tracing for observability
- Use environment variables for all secrets

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Getting Started

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes
4. **Commit** with clear messages: `git commit -m 'Add amazing feature'`
5. **Push** to your branch: `git push origin feature/amazing-feature`
6. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Write docstrings for all functions
- Add unit tests for new features
- Update documentation
- Ensure all tests pass before submitting PR

### Code Quality Tools

```bash
# Format code
black app/ tests/

# Lint code
pylint app/

# Type checking
mypy app/

# Security check
bandit -r app/
```

---

## 📚 Documentation

### Agent Architecture

Each agent implements the `BaseAgent` pattern with:
- **Input Processing**: Query/document normalization
- **Tool Invocation**: Dynamic tool selection
- **Output Formatting**: Structured response generation
- **Error Handling**: Graceful degradation

### Vector Embeddings

- **Model**: Sentence Transformers (all-MiniLM-L6-v2)
- **Database**: Qdrant (vector similarity search)
- **Chunk Size**: 512 tokens with 100-token overlap
- **Similarity Threshold**: 0.75

---

## 🔐 Security

- ✅ Environment-based secrets management
- ✅ Input validation on all endpoints
- ✅ Rate limiting via FastAPI middleware
- ✅ CORS configuration for web clients
- ✅ API authentication ready (implementable)

---

## 📈 Roadmap

- [ ] **WebSocket Support** - Real-time research updates
- [ ] **Multi-Language Support** - International research
- [ ] **Advanced Caching** - Redis integration
- [ ] **Custom Model Fine-tuning** - Domain-specific models
- [ ] **Collaborative Tools** - Team research workflows
- [ ] **Research History** - Full audit trail
- [ ] **Export Formats** - PDF, Excel, JSON exports
- [ ] **Scheduled Research** - Cron-based analysis

---

## 📞 Support & Contact

- 📧 **Email**: support@augus.dev
- 🐦 **Twitter**: [@AugusAI](https://twitter.com)
- 💬 **Discord**: [Join Community](https://discord.gg)
- 📖 **Docs**: [Full Documentation](https://docs.augus.dev)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Star History

If you find this project helpful, please consider starring it! ⭐

---

<div align="center">

### Built with ❤️ by the AUGUS Team

**[Website](https://augus.dev) • [Blog](https://blog.augus.dev) • [Twitter](https://twitter.com/AugusAI) • [GitHub](https://github.com/Code-Master-1110-PGH/Augus)**

</div>
