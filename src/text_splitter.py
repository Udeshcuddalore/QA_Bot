from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(data, chunk_size=1000, chunk_overlap=50):
    """
    Splits documents into smaller chunks for processing.

    Args:
        data (list): List of LangChain Document objects
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks

    Returns:
        list: List of split document chunks
    """
    if not data:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = splitter.split_documents(data)
    return chunks