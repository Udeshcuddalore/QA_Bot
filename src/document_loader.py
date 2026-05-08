from langchain_community.document_loaders import PyMuPDFLoader
import os
from loguru import logger

class DocumentLoader:
    def __init__(self, files):
        """
        Initializes the DocumentLoader.
        :param files: Can be a single string path, a list of paths, or file-like objects.
        """
        if not isinstance(files, list):
            self.files = [files]
        else:
            self.files = files
            
        logger.info(f"Initialized DocumentLoader with {len(self.files)} file(s)")

    def PDF_loader(self):
        all_documents = []

        for file_input in self.files:
            try:
                # 1. Determine the path
                # If it's a string, use it directly. If it's a Streamlit UploadedFile, use its name.
                if isinstance(file_input, str):
                    file_path = file_input
                elif hasattr(file_input, "name"):
                    file_path = file_input.name
                else:
                    logger.error("Input object has no name attribute and is not a string.")
                    continue

                logger.debug(f"Processing file: {file_path}")

                # 2. Validation
                if not file_path.lower().endswith(".pdf"):
                    logger.warning(f"Skipping: {file_path} is not a PDF.")
                    continue

                if not os.path.exists(file_path):
                    logger.error(f"File not found on disk: {file_path}")
                    continue

                # 3. Loading
                loader = PyMuPDFLoader(file_path)
                docs = loader.load()

                logger.info(f"Loaded {len(docs)} pages from {file_path}")
                all_documents.extend(docs)

            except Exception as e:
                logger.error(f"Error loading {file_input}: {e}")

        logger.info(f"Total pages loaded: {len(all_documents)}")
        return all_documents