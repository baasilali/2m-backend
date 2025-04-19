from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools import Tool
import os

def load_documents(directory: str = "data/documents"):
    """Load documents from a directory and create embeddings."""
    try:
        print(f"Attempting to load documents from {directory}")
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Load documents from directory
        loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        if not documents:
            print(f"No documents found in {directory}")
            return None
            
        print(f"Found {len(documents)} documents to process")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        # Create embeddings and store in FAISS
        print("Creating embeddings...")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("Document store initialized successfully")
        
        return vectorstore
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Directory contents: {os.listdir(directory) if os.path.exists(directory) else 'Directory not found'}")
        return None

def query_documents(query: str, vectorstore=None) -> str:
    """Query the document store for relevant information."""
    if vectorstore is None:
        print("Document store is None - attempting to reload documents")
        vectorstore = load_documents()
        if vectorstore is None:
            return "Document store not initialized. Please check the data/documents directory and ensure it contains .txt files."
    
    try:
        # Add CS2 context to the query if not present
        if "cs2" not in query.lower() and "counter-strike" not in query.lower():
            query = f"CS2 {query}"
        
        # Search for relevant documents with increased k for better coverage
        docs = vectorstore.similarity_search(query, k=5)
        
        # Format the results with clear CS2 context
        results = []
        for i, doc in enumerate(docs, 1):
            # Add CS2 context marker if not present
            content = doc.page_content
            if not any(marker in content.lower() for marker in ["cs2", "counter-strike", "cs:go"]):
                content = f"[CS2 Context] {content}"
            results.append(f"Document {i}:\n{content}\n")
        
        if results:
            # Check if we have complete information
            query_terms = set(query.lower().split())
            content_terms = set(" ".join(results).lower().split())
            
            # If we don't have enough matching terms, indicate incomplete information
            if len(query_terms.intersection(content_terms)) < len(query_terms) * 0.5:
                return "INCOMPLETE_INFO: Found some relevant CS2 information, but it may not be complete. Consider using web search for more details:\n\n" + "\n".join(results)
            
            return "Found relevant CS2 information in our documents:\n\n" + "\n".join(results)
        return "No relevant CS2 information found in our documents."
    except Exception as e:
        print(f"Error querying documents: {str(e)}")
        return f"Error querying documents: {str(e)}"

# Initialize the document store
print("Initializing document store...")
document_store = load_documents()

# Create the tool with updated description
document_tool = Tool(
    name="query_documents",
    func=lambda q: query_documents(q, document_store),
    description="Search through our curated CS2-specific documents for detailed information about skins, market trends, trading strategies, and price analysis. This should be used before web search or Wikipedia.",
) 