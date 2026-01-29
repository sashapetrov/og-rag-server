#!/usr/bin/env python3
"""
File Watcher for automatic RAG server updates.
Monitors document directory and automatically processes new/modified files.
"""

import os
import time
import logging
from pathlib import Path
from typing import Set, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent

logger = logging.getLogger(__name__)

class DocumentEventHandler(FileSystemEventHandler):
    """File system event handler for documents"""

    def __init__(self, processor, watch_path: str):
        """
        Initialize event handler.

        Args:
            processor: DocumentProcessor instance for file processing
            watch_path: Path to directory to watch
        """
        super().__init__()
        self.processor = processor
        self.watch_path = Path(watch_path)

        # Supported extensions
        self.supported_extensions = {'.pdf', '.md', '.txt', '.epub', '.fb2'}

        # Timer for batch processing (avoid multiple events for single file)
        self.pending_files = set()
        self.last_event_time = {}
        self.last_log_time = {}  # debounce for logs
        self.batch_delay = 2.0  # seconds to wait before processing after last event
        self.log_debounce = 1.0  # minimum interval between logs for same file

    def _should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed"""
        return (
            file_path.exists() and
            file_path.is_file() and
            file_path.suffix.lower() in self.supported_extensions and
            not file_path.name.startswith('.')  # exclude hidden files
        )

    def _debounced_log(self, file_path: Path, event_type: str) -> bool:
        """Logging with debounce - returns True if log was written"""
        key = str(file_path)
        current_time = time.time()
        last_time = self.last_log_time.get(key, 0)

        if current_time - last_time >= self.log_debounce:
            self.last_log_time[key] = current_time
            logger.info(f"File {event_type}: {file_path.name}")
            return True
        return False

    def _schedule_file_processing(self, file_path: Path):
        """Schedule file processing with delay for batch operations"""
        if not self._should_process_file(file_path):
            return

        current_time = time.time()
        self.last_event_time[str(file_path)] = current_time
        self.pending_files.add(file_path)

        logger.info(f"Scheduled processing for: {file_path.name}")

        # Start delayed processing
        def delayed_process():
            time.sleep(self.batch_delay)

            # Check if file still needs processing
            if (str(file_path) in self.last_event_time and
                self.last_event_time[str(file_path)] <= current_time + 0.1):

                if file_path in self.pending_files:
                    self.pending_files.remove(file_path)

                    if self._should_process_file(file_path):
                        try:
                            logger.info(f"Auto-processing file: {file_path.name}")
                            success = self.processor.process_file(file_path)
                            if success:
                                logger.info(f"Successfully auto-processed: {file_path.name}")
                            else:
                                logger.warning(f"Failed to auto-process: {file_path.name}")
                        except Exception as e:
                            logger.error(f"Error auto-processing {file_path}: {e}")

        # Run in separate thread
        import threading
        threading.Thread(target=delayed_process, daemon=True).start()

    def on_created(self, event):
        """Handle file creation event"""
        if not event.is_directory:
            file_path = Path(event.src_path)
            self._debounced_log(file_path, "created")
            self._schedule_file_processing(file_path)

    def on_modified(self, event):
        """Handle file modification event"""
        if not event.is_directory:
            file_path = Path(event.src_path)
            self._debounced_log(file_path, "modified")
            self._schedule_file_processing(file_path)

    def on_deleted(self, event):
        """Handle file deletion event"""
        if not event.is_directory:
            file_path = Path(event.src_path)
            logger.info(f"File deleted: {file_path.name}")

            try:
                # Remove file from collection
                success = self.processor.remove_file_from_collection(file_path)
                if success:
                    logger.info(f"Successfully removed from collection: {file_path.name}")
                else:
                    logger.warning(f"Failed to remove from collection: {file_path.name}")
            except Exception as e:
                logger.error(f"Error removing {file_path} from collection: {e}")

    def on_moved(self, event):
        """Handle file move event"""
        if not event.is_directory:
            old_path = Path(event.src_path)
            new_path = Path(event.dest_path)

            logger.info(f"File moved: {old_path.name} -> {new_path.name}")

            # Remove old file from collection
            try:
                self.processor.remove_file_from_collection(old_path)
            except Exception as e:
                logger.warning(f"Error removing old file {old_path}: {e}")

            # Add new file
            self._schedule_file_processing(new_path)

class FileWatcher:
    """Main class for file watching"""

    def __init__(self, watch_path: str, processor):
        """
        Initialize File Watcher.

        Args:
            watch_path: Path to directory to watch
            processor: DocumentProcessor instance
        """
        self.watch_path = Path(watch_path)
        self.processor = processor
        self.observer = None
        self.event_handler = DocumentEventHandler(processor, watch_path)
        self.is_running = False

        logger.info(f"FileWatcher initialized for: {self.watch_path}")

    def start_watching(self):
        """Start file watching"""
        if self.is_running:
            logger.warning("FileWatcher is already running")
            return

        if not self.watch_path.exists():
            logger.error(f"Watch path does not exist: {self.watch_path}")
            return

        try:
            self.observer = Observer()
            self.observer.schedule(
                self.event_handler,
                str(self.watch_path),
                recursive=True  # Recursively watch subdirectories
            )

            self.observer.start()
            self.is_running = True

            logger.info(f"Started watching directory: {self.watch_path}")
            logger.info("FileWatcher is running... Press Ctrl+C to stop")

            # Main loop
            try:
                while self.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop_watching()

        except Exception as e:
            logger.error(f"Failed to start FileWatcher: {e}")
            self.is_running = False

    def stop_watching(self):
        """Stop file watching"""
        if not self.is_running:
            return

        logger.info("Stopping FileWatcher...")

        if self.observer:
            self.observer.stop()
            self.observer.join()

        self.is_running = False
        logger.info("FileWatcher stopped")

    def get_status(self) -> dict:
        """Get FileWatcher status"""
        return {
            'is_running': self.is_running,
            'watch_path': str(self.watch_path),
            'supported_extensions': list(self.event_handler.supported_extensions),
            'pending_files': len(self.event_handler.pending_files)
        }

    def process_existing_files(self):
        """One-time processing of all existing files in directory"""
        logger.info("Processing existing files in watch directory...")
        self.processor.process_all_files()

class BackgroundFileWatcher:
    """Wrapper for running FileWatcher in background"""

    def __init__(self, watch_path: str, processor):
        self.file_watcher = FileWatcher(watch_path, processor)
        self.thread = None

    def start(self):
        """Start in background thread"""
        if self.thread and self.thread.is_alive():
            logger.warning("Background FileWatcher is already running")
            return

        import threading
        self.thread = threading.Thread(
            target=self.file_watcher.start_watching,
            daemon=True
        )
        self.thread.start()
        logger.info("Background FileWatcher started")

    def stop(self):
        """Stop background FileWatcher"""
        self.file_watcher.stop_watching()
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("Background FileWatcher stopped")

    def get_status(self):
        """Get status"""
        status = self.file_watcher.get_status()
        status['background_mode'] = True
        status['thread_alive'] = self.thread.is_alive() if self.thread else False
        return status
