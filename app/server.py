#!/usr/bin/env python3
"""
Autonomous RAG server with ChromaDB and automatic file watching.
"""

import os
import logging
import threading
import time
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from document_processor import DocumentProcessor
from file_watcher import FileWatcher

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class RAGServer:
    def __init__(self):
        # Paths
        self.vector_db_path = "/app/data/vector_db"
        self.documents_path = "/app/data/documents"
        self.models_path = "/app/models"

        # Create directories
        os.makedirs(self.vector_db_path, exist_ok=True)
        os.makedirs(self.documents_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)

        # Initialize ChromaDB with telemetry disabled
        self.client = chromadb.PersistentClient(
            path=self.vector_db_path,
            settings=chromadb.config.Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Create collection with cosine distance (better for text embeddings)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={
                "description": "RAG documents collection",
                "hnsw:space": "cosine"  # cosine distance: 0 = identical, 2 = opposite
            }
        )

        # Initialize document processor
        self.processor = DocumentProcessor(
            collection=self.collection,
            documents_path=self.documents_path,
            models_path=self.models_path
        )

        # Initialize file watcher
        self.file_watcher = FileWatcher(
            watch_path=self.documents_path,
            processor=self.processor
        )

        logger.info(f"RAG Server initialized. Collection size: {self.collection.count()}")

    def start_background_services(self):
        """Start background services (file watcher only)"""
        # File watcher for tracking new/modified files
        threading.Thread(
            target=self.file_watcher.start_watching,
            daemon=True
        ).start()

        logger.info("Background file watcher started")

    def search_documents(self, query: str, k: int = 3):
        """Search documents"""
        try:
            if self.collection.count() == 0:
                return []

            # Ensure embedder is loaded for search
            self.processor._ensure_embedder_loaded()

            # Create query embedding using our model
            query_embedding = self.processor.embedder.encode([query]).tolist()

            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=k
            )

            documents = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}
                    distance = results["distances"][0][i] if results["distances"] and results["distances"][0] else 0

                    # For cosine distance: 0=identical, 2=opposite
                    # cosine_similarity = 1 - distance, range [-1, 1]
                    # Normalize to [0, 1] for convenience: (1 - distance/2)
                    relevance = max(0.0, 1.0 - distance / 2.0)

                    documents.append({
                        "content": doc,
                        "metadata": metadata,
                        "distance": distance,
                        "relevance_score": relevance
                    })

            return documents

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

# Global server instance
rag_server = RAGServer()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        collection_size = rag_server.collection.count()
        return jsonify({
            'status': 'healthy',
            'service': 'autonomous-rag-server',
            'collection_size': collection_size,
            'documents_path': rag_server.documents_path,
            'vector_db_path': rag_server.vector_db_path
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/search', methods=['POST'])
def search():
    """Search documents"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        query = data.get('query', '').strip()
        k = data.get('k', 3)

        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400

        # Validate k
        try:
            k = int(k)
            if k < 1 or k > 20:
                k = 3
        except (ValueError, TypeError):
            k = 3

        logger.info(f"Search request: query='{query}', k={k}")

        results = rag_server.search_documents(query, k)

        if results:
            # Build context
            context_parts = []
            for i, result in enumerate(results, 1):
                content = result["content"]
                source = result["metadata"].get("source", "unknown")
                doc_type = result["metadata"].get("type", "unknown")
                relevance = result.get("relevance_score", 0)

                context_parts.append(
                    f"Document {i} (Type: {doc_type}, Source: {source}, Relevance: {relevance:.2f}):\n{content}"
                )

            context = "\n\n".join(context_parts)

            response = {
                'query': query,
                'results': results,
                'context': context,
                'message': f'Found {len(results)} relevant documents',
                'total_collection_size': rag_server.collection.count()
            }
        else:
            response = {
                'query': query,
                'results': [],
                'context': '',
                'message': 'No relevant documents found',
                'total_collection_size': rag_server.collection.count()
            }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Collection statistics"""
    try:
        # Get all metadata
        all_docs = rag_server.collection.get()

        stats = {
            'total_documents': len(all_docs['documents']),
            'collection_name': 'documents',
            'config': {
                'embedding_model': rag_server.processor.embedding_model,
                'chunk_size': rag_server.processor.chunk_size,
                'chunk_overlap': rag_server.processor.chunk_overlap
            },
            'types': {},
            'sources': {}
        }

        # Analyze metadata
        for metadata in all_docs['metadatas']:
            doc_type = metadata.get('type', 'unknown')
            source = metadata.get('source', 'unknown')

            stats['types'][doc_type] = stats['types'].get(doc_type, 0) + 1
            stats['sources'][source] = stats['sources'].get(source, 0) + 1

        return jsonify(stats)

    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': 'Error getting stats'}), 500

@app.route('/process', methods=['POST'])
def manual_process():
    """Manually trigger document processing"""
    try:
        threading.Thread(
            target=rag_server.processor.process_all_files,
            daemon=True
        ).start()

        return jsonify({
            'message': 'Document processing started',
            'status': 'processing'
        })

    except Exception as e:
        logger.error(f"Manual process error: {e}")
        return jsonify({'error': 'Error starting processing'}), 500

@app.route('/debug', methods=['GET'])
def debug():
    """Debug information about server state"""
    try:
        import os

        debug_info = {
            'server_status': 'running',
            'collection_size': rag_server.collection.count(),
            'documents_path': rag_server.documents_path,
            'documents_exists': os.path.exists(rag_server.documents_path),
            'files_in_documents': [],
            'processed_files_cache': len(rag_server.processor.processed_files),
            'file_watcher_status': rag_server.file_watcher.get_status() if hasattr(rag_server, 'file_watcher') else 'not_available'
        }

        # List files in documents directory
        if os.path.exists(rag_server.documents_path):
            for root, dirs, files in os.walk(rag_server.documents_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, rag_server.documents_path)
                    debug_info['files_in_documents'].append({
                        'path': relative_path,
                        'size': os.path.getsize(file_path),
                        'extension': os.path.splitext(file)[1]
                    })

        return jsonify(debug_info)

    except Exception as e:
        logger.error(f"Debug error: {e}")
        return jsonify({'error': 'Error getting debug info', 'details': str(e)}), 500


@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():
    """
    Evaluate RAG system quality

    GET: Run with auto-generated tests
    POST: Run with custom tests from body

    Body format (optional):
    {
        "test_queries": [
            {
                "query": "query text",
                "expected_sources": ["file.md"],  // optional
                "expected_keywords": ["keyword1"]  // optional
            }
        ],
        "k": 5
    }
    """
    try:
        data = request.get_json() if request.method == 'POST' else {}
        k = data.get('k', 5)

        # Custom tests or auto-generate
        test_queries = data.get('test_queries', [])

        if not test_queries:
            test_queries = _generate_generic_tests()

        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'collection_size': rag_server.collection.count(),
            'config': {
                'embedding_model': rag_server.processor.embedding_model,
                'chunk_size': rag_server.processor.chunk_size,
                'chunk_overlap': rag_server.processor.chunk_overlap
            },
            'k': k,
            'tests': [],
            'metrics': {
                'precision_at_k': 0.0,
                'mrr': 0.0,
                'avg_relevance': 0.0,
                'tests_passed': 0,
                'tests_total': 0
            }
        }

        total_precision = 0.0
        total_mrr = 0.0
        total_relevance = 0.0
        tests_with_expected = 0

        for test in test_queries:
            query = test.get('query', '')
            expected_sources = test.get('expected_sources', [])
            expected_keywords = test.get('expected_keywords', [])

            search_results = rag_server.search_documents(query, k)

            test_result = {
                'query': query,
                'results_count': len(search_results),
                'top_sources': [],
                'avg_relevance': 0.0,
                'precision': None,
                'reciprocal_rank': None
            }

            if search_results:
                relevances = []
                for i, r in enumerate(search_results):
                    source = r.get('metadata', {}).get('source', 'unknown')
                    relevance = r.get('relevance_score', 0)
                    test_result['top_sources'].append({
                        'rank': i + 1,
                        'source': source,
                        'relevance': round(relevance, 3)
                    })
                    relevances.append(relevance)

                test_result['avg_relevance'] = round(sum(relevances) / len(relevances), 3)
                total_relevance += test_result['avg_relevance']

                # Precision@k if expected_sources provided
                if expected_sources:
                    tests_with_expected += 1
                    found_sources = [s['source'] for s in test_result['top_sources']]
                    matches = sum(1 for es in expected_sources if any(es.lower() in fs.lower() for fs in found_sources))
                    test_result['precision'] = round(matches / len(expected_sources), 3)
                    total_precision += test_result['precision']

                    # Reciprocal Rank
                    for i, fs in enumerate(found_sources):
                        if any(es.lower() in fs.lower() for es in expected_sources):
                            test_result['reciprocal_rank'] = round(1.0 / (i + 1), 3)
                            total_mrr += test_result['reciprocal_rank']
                            break
                    else:
                        test_result['reciprocal_rank'] = 0.0

                # Keyword check
                if expected_keywords:
                    found_text = ' '.join([r.get('content', '') for r in search_results]).lower()
                    keywords_found = sum(1 for kw in expected_keywords if kw.lower() in found_text)
                    test_result['keywords_found'] = f"{keywords_found}/{len(expected_keywords)}"

            results['tests'].append(test_result)

        # Final metrics
        n_tests = len(test_queries)
        if n_tests > 0:
            results['metrics']['avg_relevance'] = round(total_relevance / n_tests, 3)
            results['metrics']['tests_total'] = n_tests
            results['metrics']['tests_passed'] = sum(1 for t in results['tests'] if t['results_count'] > 0)

            if tests_with_expected > 0:
                results['metrics']['precision_at_k'] = round(total_precision / tests_with_expected, 3)
                results['metrics']['mrr'] = round(total_mrr / tests_with_expected, 3)

        return jsonify(results)

    except Exception as e:
        logger.error(f"Evaluate error: {e}")
        return jsonify({'error': str(e)}), 500


def _generate_generic_tests():
    """Generate tests based on existing documents"""
    tests = []

    try:
        all_docs = rag_server.collection.get()

        if all_docs['metadatas']:
            sources = set()
            for metadata in all_docs['metadatas']:
                source = metadata.get('source', '')
                if source:
                    sources.add(source)

            # Generate tests from filenames (max 5)
            for source in list(sources)[:5]:
                # Extract words from filename
                filename = source.replace('/', ' ').replace('_', ' ').replace('-', ' ')
                filename = filename.rsplit('.', 1)[0]  # Remove extension

                # Take first 3-4 words
                words = [w for w in filename.split() if len(w) > 2][:4]
                if words:
                    tests.append({
                        'query': ' '.join(words),
                        'expected_sources': [source]
                    })

        if not tests:
            tests = [
                {'query': 'documentation'},
                {'query': 'configuration'},
                {'query': 'installation'}
            ]

    except Exception as e:
        logger.warning(f"Failed to generate generic tests: {e}")
        tests = [{'query': 'test query'}]

    return tests


if __name__ == '__main__':
    # 1. Process all existing documents (synchronously, before HTTP server starts)
    logger.info("Processing all documents on startup...")
    try:
        rag_server.processor.process_all_files()
        logger.info(f"Initial processing completed. Collection size: {rag_server.collection.count()}")
    except Exception as e:
        logger.error(f"Error during initial document processing: {e}")

    # 2. Start file watcher for tracking new files
    rag_server.start_background_services()

    logger.info("Starting RAG HTTP server on 0.0.0.0:5000...")

    # 3. Start Flask server
    app.run(
        host='0.0.0.0',
        port=5000,
        threaded=True,
        debug=False
    )
