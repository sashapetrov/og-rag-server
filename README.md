# OG RAG Server

[![Build and Push](https://github.com/cummunder/og-rag-server/actions/workflows/docker-build.yml/badge.svg)](https://github.com/cummunder/og-rag-server/actions/workflows/docker-build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker Image](https://img.shields.io/badge/ghcr.io-og--rag--server-blue)](https://github.com/cummunder/og-rag-server/pkgs/container/og-rag-server)

Autonomous RAG server (Retrieval Augmented Generation) with ChromaDB, automatic file watching, and configurable embeddings.

## Features

- Automatic document directory monitoring (file watcher)
- Supported formats: PDF, Markdown, TXT, EPUB, FB2
- Configurable embedding model
- Configurable chunking parameters
- REST API for search and management
- Quality evaluation endpoint `/evaluate`
- Persistent storage via Docker volumes

## Installation

### Using pre-built image (recommended)

```bash
docker pull ghcr.io/cummunder/og-rag-server:latest
```

### Build from source

```bash
git clone https://github.com/cummunder/og-rag-server.git
cd og-rag-server
docker build -t og-rag-server .
```

## Quick Start

```bash
# Using pre-built image
docker run -d -p 45000:5000 \
  -v ./documents:/app/data/documents \
  -v rag_data:/app/data \
  ghcr.io/cummunder/og-rag-server:latest

# Or build and run locally
docker-compose up -d

# Add documents
cp your_documents/* ./documents/

# Health check
curl http://localhost:45000/health

# Search
curl -X POST http://localhost:45000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your query", "k": 3}'
```

## Environment Variables

### Embedding Model

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model for embeddings |

**Recommended models:**

| Model | Languages | Dimensions | Speed | Use Case |
|-------|-----------|------------|-------|----------|
| `all-MiniLM-L6-v2` | EN | 384 | Fast | English technical docs |
| `intfloat/multilingual-e5-base` | 50+ | 768 | Medium | Mixed multilingual content |
| `paraphrase-multilingual-MiniLM-L12-v2` | 50+ | 384 | Fast | Multilingual balanced |
| `cointegrated/rubert-tiny2` | RU | 312 | Fastest | Russian-only content |

### Chunking Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `1000` | Chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |

**Recommendations by content type:**

| Content Type | CHUNK_SIZE | CHUNK_OVERLAP | Description |
|--------------|------------|---------------|-------------|
| Technical documentation | 800 | 150 | Short precise answers |
| Books / long-form content | 1500 | 300 | Preserve context |
| FAQ / short articles | 500 | 100 | Compact answers |

### System Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKUP_ENABLED` | `false` | Enable automatic backup |
| `BACKUP_RETENTION_DAYS` | `1` | Backup retention days |
| `BACKUP_INTERVAL_HOURS` | `6` | Backup interval |
| `AUTONOMOUS_MODE` | `true` | Autonomous mode |

## API Endpoints

### POST /search

Search documents by query.

```bash
curl -X POST http://localhost:45000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "kubernetes deployment", "k": 5}'
```

**Response:**
```json
{
  "query": "kubernetes deployment",
  "results": [...],
  "context": "...",
  "message": "Found 5 relevant documents",
  "total_collection_size": 35220
}
```

### GET /health

Service health check.

```bash
curl http://localhost:45000/health
```

### GET /stats

Collection statistics with configuration.

```bash
curl http://localhost:45000/stats
```

**Response:**
```json
{
  "total_documents": 35220,
  "collection_name": "documents",
  "config": {
    "embedding_model": "all-MiniLM-L6-v2",
    "chunk_size": 1000,
    "chunk_overlap": 200
  },
  "types": {"pdf": 5563, "fb2": 29047, "epub": 610},
  "sources": {...}
}
```

### GET /debug

Debug information about server state.

### POST /process

Manually trigger document reindexing.

```bash
curl -X POST http://localhost:45000/process
```

### GET|POST /evaluate

Evaluate RAG system quality.

```bash
# Auto-generated tests based on existing documents
curl http://localhost:45000/evaluate

# Custom tests
curl -X POST http://localhost:45000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "test_queries": [
      {
        "query": "how to configure kubernetes",
        "expected_sources": ["k8s-guide.md"],
        "expected_keywords": ["kubectl", "deployment"]
      }
    ],
    "k": 5
  }'
```

**Response:**
```json
{
  "timestamp": "2025-01-29 12:00:00",
  "collection_size": 35220,
  "config": {
    "embedding_model": "all-MiniLM-L6-v2",
    "chunk_size": 1000,
    "chunk_overlap": 200
  },
  "metrics": {
    "precision_at_k": 0.8,
    "mrr": 0.75,
    "avg_relevance": 0.45,
    "tests_passed": 5,
    "tests_total": 5
  },
  "tests": [...]
}
```

**Metrics:**
- `precision_at_k` - fraction of queries where expected source found in top-k
- `mrr` - Mean Reciprocal Rank (average reciprocal rank of correct answer)
- `avg_relevance` - average relevance score of results

## Docker Compose Examples

### Russian Book Library

```yaml
services:
  rag-books-ru:
    image: og-rag-server:latest
    ports:
      - "45100:5000"
    environment:
      - EMBEDDING_MODEL=cointegrated/rubert-tiny2
      - CHUNK_SIZE=1500
      - CHUNK_OVERLAP=300
    volumes:
      - ./books:/app/data/documents
      - rag_books_data:/app/data
```

### Technical Documentation (EN)

```yaml
services:
  rag-techdocs:
    image: og-rag-server:latest
    ports:
      - "45000:5000"
    environment:
      - EMBEDDING_MODEL=all-MiniLM-L6-v2
      - CHUNK_SIZE=800
      - CHUNK_OVERLAP=150
    volumes:
      - ./docs:/app/data/documents
      - rag_docs_data:/app/data
```

### Mixed Content (Multilingual)

```yaml
services:
  rag-mixed:
    image: og-rag-server:latest
    ports:
      - "45200:5000"
    environment:
      - EMBEDDING_MODEL=intfloat/multilingual-e5-base
      - CHUNK_SIZE=1000
      - CHUNK_OVERLAP=200
    volumes:
      - ./mixed-docs:/app/data/documents
      - rag_mixed_data:/app/data
```

## Migration and Reindexing

### Important: Update to Cosine Distance

This version uses **cosine distance** instead of L2 (Euclidean).
This provides more accurate relevance scores for text embeddings.

**For existing installations**, reindexing is required:

```bash
docker-compose down
docker volume rm rag_enhanced_persistent_data
docker-compose up -d
```

### Changing Embedding Model

When changing `EMBEDDING_MODEL`, full reindexing is required:

```bash
# 1. Stop container
docker-compose down

# 2. Remove old vector database (documents are preserved!)
docker volume rm rag_enhanced_persistent_data

# 3. Change EMBEDDING_MODEL in docker-compose.yaml

# 4. Restart (reindexing is automatic)
docker-compose up -d

# 5. Monitor logs
docker-compose logs -f
```

### Changing Chunk Parameters

Changes to `CHUNK_SIZE` or `CHUNK_OVERLAP` only affect new files. For full reindexing:

```bash
docker-compose down
docker volume rm rag_enhanced_persistent_data
docker-compose up -d
```

## Supported Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| PDF | `.pdf` | PDF documents |
| Markdown | `.md` | Markdown files |
| Text | `.txt` | Plain text files |
| EPUB | `.epub` | E-books |
| FictionBook | `.fb2` | FB2 format |

## Architecture

```
og-rag-server/
├── app/
│   ├── server.py              # Flask HTTP API
│   ├── document_processor.py  # Document processing + embeddings
│   └── file_watcher.py        # File monitoring (watchdog)
├── config/
│   └── settings.py            # Configuration (env vars)
├── scripts/
│   ├── entrypoint.sh          # Docker entrypoint
│   └── backup_data.sh         # Backup script
├── docker-compose.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Troubleshooting

### Poor Search Quality for Non-English Content

Switch to a multilingual model:
```yaml
- EMBEDDING_MODEL=intfloat/multilingual-e5-base
```

### Results Break Context

Increase chunk size:
```yaml
- CHUNK_SIZE=1500
- CHUNK_OVERLAP=300
```

### Slow First Load

On first startup, the model downloads from HuggingFace. This is normal. After download, the model is cached in the volume.

### Verify Configuration

```bash
# Check current configuration
curl http://localhost:45000/stats | jq '.config'

# Evaluate quality
curl http://localhost:45000/evaluate | jq '.metrics'
```

## License

MIT
