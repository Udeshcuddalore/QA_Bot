from langchain_community.document_loaders import PyPDFLoader
import os
from loguru import logger

class DocumentLoader:
    def __init__(self, files):
        # Ensure it's always a list
        if not isinstance(files, list):
            files = [files]

        self.files = files
        logger.info(f"Initialized DocumentLoader with {len(self.files)} file(s)")

    def PDF_loader(self):
        all_documents = []

        for file in self.files:
            try:
                logger.debug(f"Processing file: {getattr(file, 'name', 'unknown')}")

                if not hasattr(file, "name"):
                    raise ValueError("Invalid file object")

                if not file.name.lower().endswith(".pdf"):
                    raise ValueError(f"Not a PDF: {file.name}")

                if isinstance(file.name, str) and not os.path.exists(file.name):
                    raise FileNotFoundError(f"File not found: {file.name}")

                loader = PyPDFLoader(file.name)
                docs = loader.load()

                logger.info(f"Loaded {len(docs)} documents from {file.name}")
                all_documents.extend(docs)

            except Exception as e:
                logger.error(
                    f"Skipping {getattr(file, 'name', 'unknown file')} due to error: {e}"
                )

        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents



    