#!/usr/bin/env python3
"""
Document processor for autonomous RAG server.
Supports PDF, Markdown, TXT, EPUB, FB2 files with configurable embeddings.
"""

import os
import sys
import logging
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional, Union

import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import markdown
import ebooklib
from ebooklib import epub
from lxml import etree

# Import configuration from settings
sys.path.insert(0, '/app/config')
try:
    from settings import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
except ImportError:
    # Fallback defaults if settings not available
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

logger = logging.getLogger(__name__)

# ChromaDB max batch size (default 5461, use conservative value)
CHROMA_MAX_BATCH_SIZE = 5000

# CPU throttling: delay between batches (seconds) to prevent CPU overload
BATCH_PROCESSING_DELAY = 0.1

# Large file threshold for progressive processing
LARGE_FILE_CHUNK_THRESHOLD = 2000


class DocumentProcessor:
    def __init__(self, collection, documents_path: str, models_path: str,
                 embedding_model: str = None, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize document processor.

        Args:
            collection: ChromaDB collection for storing vectors
            documents_path: Path to documents directory
            models_path: Path to models cache
            embedding_model: Embedding model (default from settings/env)
            chunk_size: Chunk size in characters (default from settings/env)
            chunk_overlap: Chunk overlap in characters (default from settings/env)
        """
        self.collection = collection
        self.documents_path = Path(documents_path)
        self.models_path = Path(models_path)

        # Configurable parameters with defaults from settings
        self.embedding_model = embedding_model or EMBEDDING_MODEL
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP

        # Lazy model initialization (loaded on first use)
        self.embedder = None

        logger.info(f"DocumentProcessor initialized: model={self.embedding_model}, "
                    f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")

        # Supported extensions
        self.supported_extensions = {'.pdf', '.md', '.txt', '.epub', '.fb2'}

        # Processed files cache (hash -> metadata)
        self.processed_files = {}
        self._load_processed_cache()

    def _ensure_embedder_loaded(self):
        """Load embedding model on first use"""
        if self.embedder is None:
            try:
                logger.info(f"Loading embedding model: {self.embedding_model}...")
                # Force online mode for model download
                os.environ['TRANSFORMERS_OFFLINE'] = '0'
                os.environ['HF_DATASETS_OFFLINE'] = '0'

                self.embedder = SentenceTransformer(
                    self.embedding_model,
                    cache_folder=str(self.models_path)
                )
                logger.info(f"Embedding model '{self.embedding_model}' loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model '{self.embedding_model}': {e}")
                raise

    def _load_processed_cache(self):
        """Load processed files cache from collection metadata"""
        try:
            all_docs = self.collection.get()
            if all_docs['metadatas']:
                for metadata in all_docs['metadatas']:
                    file_hash = metadata.get('file_hash')
                    if file_hash:
                        self.processed_files[file_hash] = metadata
            logger.info(f"Loaded {len(self.processed_files)} processed files from cache")
        except Exception as e:
            logger.warning(f"Failed to load processed files cache: {e}")

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate file hash for change detection"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""

    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            reader = PdfReader(str(file_path))
            text_parts = []

            for page_num, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text()
                    if text.strip():
                        text_parts.append(f"[Page {page_num}]\n{text}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num} in {file_path}: {e}")
                    continue

            full_text = "\n\n".join(text_parts)
            logger.info(f"Extracted text from PDF {file_path.name}: {len(full_text)} characters")
            return full_text

        except Exception as e:
            logger.error(f"Failed to extract text from PDF {file_path}: {e}")
            return ""

    def _extract_text_from_markdown(self, file_path: Path) -> str:
        """Extract text from Markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Convert Markdown to plain text
            html = markdown.markdown(content)
            # Simple HTML tag removal
            import re
            text = re.sub('<[^<]+?>', '', html)

            logger.info(f"Extracted text from Markdown {file_path.name}: {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Failed to extract text from Markdown {file_path}: {e}")
            return ""

    def _extract_text_from_txt(self, file_path: Path) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            logger.info(f"Extracted text from TXT {file_path.name}: {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Failed to extract text from TXT {file_path}: {e}")
            return ""

    def _extract_text_from_epub(self, file_path: Path) -> str:
        """Extract text from EPUB file"""
        try:
            book = epub.read_epub(str(file_path))
            text_parts = []

            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    try:
                        content = item.get_content()
                        # Parse HTML content
                        tree = etree.HTML(content)
                        if tree is not None:
                            # Extract all text
                            text = ' '.join(tree.xpath('//text()'))
                            text = ' '.join(text.split())  # Normalize whitespace
                            if text.strip():
                                text_parts.append(text)
                    except Exception as e:
                        logger.warning(f"Failed to extract text from EPUB item: {e}")
                        continue

            full_text = "\n\n".join(text_parts)
            logger.info(f"Extracted text from EPUB {file_path.name}: {len(full_text)} characters")
            return full_text

        except Exception as e:
            logger.error(f"Failed to extract text from EPUB {file_path}: {e}")
            return ""

    def _extract_text_from_fb2(self, file_path: Path) -> str:
        """Extract text from FB2 file"""
        try:
            # FB2 is an XML format
            with open(file_path, 'rb') as f:
                content = f.read()

            # Parse XML
            tree = etree.fromstring(content)

            # FB2 namespace
            ns = {'fb': 'http://www.gribuser.ru/xml/fictionbook/2.0'}

            text_parts = []

            # Extract book title
            title_info = tree.xpath('//fb:title-info/fb:book-title/text()', namespaces=ns)
            if title_info:
                text_parts.append(f"Title: {title_info[0]}")

            # Extract author
            authors = tree.xpath('//fb:title-info/fb:author', namespaces=ns)
            for author in authors:
                first_name = author.xpath('fb:first-name/text()', namespaces=ns)
                last_name = author.xpath('fb:last-name/text()', namespaces=ns)
                author_name = ' '.join(filter(None, [
                    first_name[0] if first_name else '',
                    last_name[0] if last_name else ''
                ]))
                if author_name.strip():
                    text_parts.append(f"Author: {author_name}")

            # Extract annotation
            annotation = tree.xpath('//fb:title-info/fb:annotation//text()', namespaces=ns)
            if annotation:
                ann_text = ' '.join(' '.join(annotation).split())
                if ann_text.strip():
                    text_parts.append(f"Annotation: {ann_text}")

            # Extract main text from body
            body_texts = tree.xpath('//fb:body//text()', namespaces=ns)
            body_text = ' '.join(' '.join(body_texts).split())
            if body_text.strip():
                text_parts.append(body_text)

            full_text = "\n\n".join(text_parts)
            logger.info(f"Extracted text from FB2 {file_path.name}: {len(full_text)} characters")
            return full_text

        except Exception as e:
            logger.error(f"Failed to extract text from FB2 {file_path}: {e}")
            return ""

    def _split_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """Split text into chunks for vectorization"""
        if not text.strip():
            return []

        # Use instance parameters if not explicitly passed
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)

            # Try to find sentence boundary within chunk
            if end < text_length:
                # Look for nearest period or newline
                for boundary in ['.', '\n', '!', '?']:
                    boundary_pos = text.rfind(boundary, start, end)
                    if boundary_pos > start + chunk_size // 2:
                        end = boundary_pos + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = max(start + chunk_size - chunk_overlap, end)

        logger.info(f"Split text into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
        return chunks

    def process_file(self, file_path: Path) -> bool:
        """
        Process a single file.

        Args:
            file_path: Path to file to process

        Returns:
            bool: True if file was successfully processed
        """
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return False

        if file_path.suffix.lower() not in self.supported_extensions:
            logger.info(f"Skipping unsupported file: {file_path}")
            return False

        # Check if file needs reprocessing
        current_hash = self._calculate_file_hash(file_path)
        if not current_hash:
            return False

        # Check cache
        if current_hash in self.processed_files:
            logger.info(f"File {file_path.name} already processed (hash: {current_hash[:8]})")
            return True

        logger.info(f"Processing file: {file_path.name}")

        # Extract text based on file type
        suffix = file_path.suffix.lower()
        if suffix == '.pdf':
            text = self._extract_text_from_pdf(file_path)
        elif suffix == '.md':
            text = self._extract_text_from_markdown(file_path)
        elif suffix == '.txt':
            text = self._extract_text_from_txt(file_path)
        elif suffix == '.epub':
            text = self._extract_text_from_epub(file_path)
        elif suffix == '.fb2':
            text = self._extract_text_from_fb2(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path.suffix}")
            return False

        if not text.strip():
            logger.warning(f"No text extracted from {file_path}")
            return False

        # Split into chunks
        chunks = self._split_text(text)
        if not chunks:
            logger.warning(f"No chunks created from {file_path}")
            return False

        try:
            # Ensure model is loaded
            self._ensure_embedder_loaded()

            # Generate metadata
            base_metadata = {
                'source': str(file_path.relative_to(self.documents_path)),
                'file_hash': current_hash,
                'type': file_path.suffix.lower()[1:],  # without dot
                'total_chunks': len(chunks),
                'file_size': file_path.stat().st_size,
                'processed_at': str(Path().resolve())
            }

            # Add chunks to collection
            chunk_ids = []
            chunk_metadatas = []

            for i, chunk in enumerate(chunks):
                chunk_id = f"{current_hash}_{i}"
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_index': i,
                    'chunk_id': chunk_id
                })

                chunk_ids.append(chunk_id)
                chunk_metadatas.append(chunk_metadata)

            # Add to ChromaDB with batching for large files
            total_chunks = len(chunks)
            if total_chunks > CHROMA_MAX_BATCH_SIZE:
                logger.info(f"Large file detected: {total_chunks} chunks, processing in batches of {CHROMA_MAX_BATCH_SIZE}")

            for batch_start in range(0, total_chunks, CHROMA_MAX_BATCH_SIZE):
                batch_end = min(batch_start + CHROMA_MAX_BATCH_SIZE, total_chunks)
                batch_chunks = chunks[batch_start:batch_end]
                batch_metadatas = chunk_metadatas[batch_start:batch_end]
                batch_ids = chunk_ids[batch_start:batch_end]

                # Create embeddings using our model
                batch_embeddings = self.embedder.encode(batch_chunks).tolist()

                self.collection.add(
                    documents=batch_chunks,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )

                if total_chunks > CHROMA_MAX_BATCH_SIZE:
                    logger.info(f"Batch {batch_start // CHROMA_MAX_BATCH_SIZE + 1}: added {len(batch_chunks)} chunks ({batch_end}/{total_chunks})")
                    # CPU throttling: small delay between batches to prevent overload
                    time.sleep(BATCH_PROCESSING_DELAY)

            # Update cache
            self.processed_files[current_hash] = base_metadata

            logger.info(f"Successfully processed {file_path.name}: {len(chunks)} chunks added")
            return True

        except Exception as e:
            logger.error(f"Failed to add chunks to collection for {file_path}: {e}")
            return False

    def process_all_files(self):
        """Process all files in documents directory"""
        if not self.documents_path.exists():
            logger.warning(f"Documents directory not found: {self.documents_path}")
            return

        logger.info(f"Processing all files in {self.documents_path}")

        processed_count = 0
        skipped_count = 0
        error_count = 0

        # Recursive search for all supported files
        for file_path in self.documents_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    if self.process_file(file_path):
                        processed_count += 1
                        # CPU throttling between files
                        time.sleep(BATCH_PROCESSING_DELAY)
                    else:
                        skipped_count += 1
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    error_count += 1

        logger.info(f"Processing complete: {processed_count} processed, {skipped_count} skipped, {error_count} errors")

    def remove_file_from_collection(self, file_path: Path):
        """Remove file from collection (when file is deleted)"""
        try:
            # Calculate file hash if it still exists
            if file_path.exists():
                file_hash = self._calculate_file_hash(file_path)
            else:
                # Search by relative path in metadata
                relative_path = str(file_path.relative_to(self.documents_path))
                file_hash = None
                for hash_key, metadata in self.processed_files.items():
                    if metadata.get('source') == relative_path:
                        file_hash = hash_key
                        break

            if not file_hash:
                logger.warning(f"Could not find hash for file {file_path}")
                return False

            # Get all chunks for this file
            all_docs = self.collection.get()
            ids_to_delete = []

            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata and metadata.get('file_hash') == file_hash:
                    ids_to_delete.append(all_docs['ids'][i])

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Removed {len(ids_to_delete)} chunks for file {file_path.name}")

            # Remove from cache
            if file_hash in self.processed_files:
                del self.processed_files[file_hash]

            return True

        except Exception as e:
            logger.error(f"Failed to remove file {file_path} from collection: {e}")
            return False

    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'processed_files': len(self.processed_files),
                'supported_extensions': list(self.supported_extensions),
                'embedding_model': self.embedding_model,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
