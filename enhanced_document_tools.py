import os
import json
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools import Tool
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDocumentProcessor:
    """
    Enhanced document processing system that combines multiple embedding approaches
    for better CS2 information retrieval.
    """
    
    def __init__(self, documents_dir: str = "data/documents", 
                 use_openai: bool = True, 
                 use_local_embeddings: bool = True):
        """
        Initialize the enhanced document processor
        
        Args:
            documents_dir: Directory containing CS2 documents
            use_openai: Whether to use OpenAI embeddings
            use_local_embeddings: Whether to use local sentence transformers
        """
        self.documents_dir = documents_dir
        self.use_openai = use_openai
        self.use_local_embeddings = use_local_embeddings
        
        # Vector stores
        self.openai_vectorstore = None
        self.local_vectorstore = None
        
        # Local embedding model
        self.local_model = None
        self.local_embeddings = None
        self.document_chunks = []
        
        # Metadata indices
        self.topic_index = {}
        self.strategy_index = {}
        self.price_analysis_index = {}
        
        # Cache paths
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "doc_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize the system
        self._initialize()
    
    def _initialize(self):
        """Initialize the document processing system"""
        logger.info("Initializing enhanced document processor...")
        
        # Load local embedding model if needed
        if self.use_local_embeddings:
            try:
                self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded local embedding model")
            except Exception as e:
                logger.error(f"Error loading local model: {e}")
                self.use_local_embeddings = False
        
        # Load and process documents
        self._load_and_process_documents()
    
    def _load_and_process_documents(self):
        """Load documents and create multiple vector representations"""
        try:
            logger.info(f"Loading documents from {self.documents_dir}")
            
            # Create directory if it doesn't exist
            os.makedirs(self.documents_dir, exist_ok=True)
            
            # Load documents
            loader = DirectoryLoader(
                self.documents_dir, 
                glob="**/*.txt", 
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            documents = loader.load()
            
            if not documents:
                logger.warning(f"No documents found in {self.documents_dir}")
                return
            
            logger.info(f"Found {len(documents)} documents to process")
            
            # Enhanced text splitting with context preservation
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Smaller chunks for better precision
                chunk_overlap=200,  # Larger overlap for context
                length_function=len,
                separators=["\n\n", "\n", ". ", ", ", " ", ""]
            )
            
            # Split documents and enhance with metadata
            self.document_chunks = []
            for doc in documents:
                chunks = text_splitter.split_documents([doc])
                
                # Enhance each chunk with CS2-specific metadata
                for chunk in chunks:
                    enhanced_chunk = self._enhance_chunk_metadata(chunk)
                    self.document_chunks.append(enhanced_chunk)
            
            logger.info(f"Created {len(self.document_chunks)} enhanced chunks")
            
            # Create vector stores
            if self.use_openai:
                self._create_openai_vectorstore()
            
            if self.use_local_embeddings:
                self._create_local_vectorstore()
            
            # Build topical indices
            self._build_topical_indices()
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
    
    def _enhance_chunk_metadata(self, chunk: Document) -> Document:
        """Enhance document chunks with CS2-specific metadata"""
        content = chunk.page_content
        content_lower = content.lower()
        
        # Extract CS2-specific topics
        topics = []
        strategies = []
        
        # Weapon-related topics
        if any(weapon in content_lower for weapon in ['ak-47', 'ak47', 'm4a4', 'm4a1-s', 'awp']):
            topics.append('weapons')
        
        # Trading topics
        if any(term in content_lower for term in ['trading', 'marketplace', 'price', 'sell', 'buy']):
            topics.append('trading')
        
        # Market analysis topics
        if any(term in content_lower for term in ['analysis', 'trend', 'market', 'value', 'investment']):
            topics.append('market_analysis')
        
        # Pattern recognition topics
        if any(term in content_lower for term in ['pattern', 'fade', 'doppler', 'case hardened']):
            topics.append('patterns')
        
        # Strategy topics
        if any(term in content_lower for term in ['strategy', 'tips', 'guide', 'how to', 'best']):
            strategies.append('guide')
        
        if any(term in content_lower for term in ['profit', 'investment', 'roi', 'return']):
            strategies.append('investment')
        
        # Create enhanced content for better embedding
        enhanced_content = content
        
        # Add contextual markers for better semantic understanding
        if topics:
            enhanced_content = f"[CS2 Topics: {', '.join(topics)}] {enhanced_content}"
        
        if strategies:
            enhanced_content = f"[Strategy: {', '.join(strategies)}] {enhanced_content}"
        
        # Update chunk metadata
        chunk.metadata.update({
            'topics': topics,
            'strategies': strategies,
            'enhanced_content': enhanced_content,
            'chunk_type': 'cs2_document'
        })
        
        # Use enhanced content for the chunk
        chunk.page_content = enhanced_content
        
        return chunk
    
    def _create_openai_vectorstore(self):
        """Create OpenAI-based vector store"""
        try:
            logger.info("Creating OpenAI vector store...")
            embeddings = OpenAIEmbeddings()
            self.openai_vectorstore = FAISS.from_documents(self.document_chunks, embeddings)
            logger.info("OpenAI vector store created successfully")
        except Exception as e:
            logger.error(f"Error creating OpenAI vector store: {e}")
            self.use_openai = False
    
    def _create_local_vectorstore(self):
        """Create local embedding-based vector store"""
        try:
            logger.info("Creating local vector store...")
            
            # Extract texts for embedding
            texts = [chunk.page_content for chunk in self.document_chunks]
            
            # Generate embeddings
            self.local_embeddings = self.local_model.encode(texts, show_progress_bar=True)
            
            logger.info("Local vector store created successfully")
        except Exception as e:
            logger.error(f"Error creating local vector store: {e}")
            self.use_local_embeddings = False
    
    def _build_topical_indices(self):
        """Build topic-based indices for faster filtering"""
        for i, chunk in enumerate(self.document_chunks):
            topics = chunk.metadata.get('topics', [])
            strategies = chunk.metadata.get('strategies', [])
            
            # Build topic index
            for topic in topics:
                if topic not in self.topic_index:
                    self.topic_index[topic] = []
                self.topic_index[topic].append(i)
            
            # Build strategy index
            for strategy in strategies:
                if strategy not in self.strategy_index:
                    self.strategy_index[strategy] = []
                self.strategy_index[strategy].append(i)
    
    def query_documents(self, query: str, max_results: int = 5, 
                       filter_topic: Optional[str] = None) -> str:
        """
        Enhanced document querying with multiple embedding approaches
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            filter_topic: Optional topic filter
            
        Returns:
            Formatted string with relevant document excerpts
        """
        try:
            # Classify query intent
            intent = self._classify_query_intent(query)
            
            # Get results from multiple sources
            results = []
            
            # OpenAI embeddings results
            if self.use_openai and self.openai_vectorstore:
                openai_results = self._query_openai_vectorstore(query, max_results, intent)
                results.extend(openai_results)
            
            # Local embeddings results
            if self.use_local_embeddings and self.local_embeddings is not None:
                local_results = self._query_local_vectorstore(query, max_results, intent)
                results.extend(local_results)
            
            # Deduplicate and rank results
            unique_results = self._deduplicate_and_rank_results(results, query)
            
            # Apply topic filtering if specified
            if filter_topic:
                unique_results = self._filter_by_topic(unique_results, filter_topic)
            
            # Format results
            return self._format_document_results(unique_results[:max_results], query, intent)
            
        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            return f"Error retrieving document information: {str(e)}"
    
    def _classify_query_intent(self, query: str) -> Dict[str, Any]:
        """Classify the intent of a document query"""
        query_lower = query.lower()
        intent = {
            'topics': [],
            'query_type': 'general',
            'specific_weapons': [],
            'specific_concepts': []
        }
        
        # Detect weapon-specific queries
        weapons = ['ak-47', 'ak47', 'm4a4', 'm4a1-s', 'awp', 'glock', 'usp', 'knife']
        intent['specific_weapons'] = [w for w in weapons if w in query_lower]
        
        # Detect topic areas
        if any(term in query_lower for term in ['price', 'cost', 'value', 'expensive', 'cheap']):
            intent['topics'].append('pricing')
            
        if any(term in query_lower for term in ['trade', 'trading', 'marketplace', 'sell', 'buy']):
            intent['topics'].append('trading')
            
        if any(term in query_lower for term in ['pattern', 'fade', 'doppler', 'blue gem']):
            intent['topics'].append('patterns')
            
        if any(term in query_lower for term in ['strategy', 'guide', 'how to', 'tips']):
            intent['query_type'] = 'guide'
            
        if any(term in query_lower for term in ['analysis', 'trend', 'market']):
            intent['query_type'] = 'analysis'
        
        return intent
    
    def _query_openai_vectorstore(self, query: str, max_results: int, 
                                 intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query OpenAI vector store"""
        try:
            enhanced_query = self._enhance_query_for_documents(query, intent)
            docs = self.openai_vectorstore.similarity_search(enhanced_query, k=max_results)
            
            results = []
            for doc in docs:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source': 'openai',
                    'score': 1.0  # FAISS doesn't return scores directly
                })
            
            return results
        except Exception as e:
            logger.error(f"Error querying OpenAI vectorstore: {e}")
            return []
    
    def _query_local_vectorstore(self, query: str, max_results: int, 
                                intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query local embedding vector store"""
        try:
            enhanced_query = self._enhance_query_for_documents(query, intent)
            
            # Generate query embedding
            query_embedding = self.local_model.encode([enhanced_query])
            
            # Calculate similarities
            similarities = np.dot(self.local_embeddings, query_embedding.T).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:max_results]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Threshold for relevance
                    chunk = self.document_chunks[idx]
                    results.append({
                        'content': chunk.page_content,
                        'metadata': chunk.metadata,
                        'source': 'local',
                        'score': float(similarities[idx])
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error querying local vectorstore: {e}")
            return []
    
    def _enhance_query_for_documents(self, query: str, intent: Dict[str, Any]) -> str:
        """Enhance query with CS2 context for better document matching"""
        enhanced_parts = [query]
        
        # Add CS2 context if not present
        if not any(term in query.lower() for term in ['cs2', 'counter-strike', 'csgo']):
            enhanced_parts.append("CS2 Counter-Strike")
        
        # Add topic-specific context
        if 'pricing' in intent['topics']:
            enhanced_parts.append("price analysis market value")
        
        if 'trading' in intent['topics']:
            enhanced_parts.append("trading marketplace strategy")
        
        if 'patterns' in intent['topics']:
            enhanced_parts.append("skin patterns quality")
        
        # Add weapon-specific context
        if intent['specific_weapons']:
            enhanced_parts.extend(intent['specific_weapons'])
        
        return " ".join(enhanced_parts)
    
    def _deduplicate_and_rank_results(self, results: List[Dict[str, Any]], 
                                     query: str) -> List[Dict[str, Any]]:
        """Deduplicate and rank results from multiple sources"""
        # Simple deduplication based on content similarity
        unique_results = []
        seen_content = set()
        
        # Sort by score first
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        for result in results:
            content = result['content']
            # Use first 100 characters as a simple deduplication key
            content_key = content[:100].strip()
            
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)
        
        return unique_results
    
    def _filter_by_topic(self, results: List[Dict[str, Any]], 
                        topic: str) -> List[Dict[str, Any]]:
        """Filter results by topic"""
        filtered = []
        for result in results:
            topics = result.get('metadata', {}).get('topics', [])
            if topic in topics:
                filtered.append(result)
        return filtered
    
    def _format_document_results(self, results: List[Dict[str, Any]], 
                                query: str, intent: Dict[str, Any]) -> str:
        """Format document results for display"""
        if not results:
            return f"No relevant CS2 documentation found for '{query}'. The system searched through available guides and analysis documents."
        
        output = []
        output.append(f"Found {len(results)} relevant CS2 document sections for '{query}':\n")
        
        for i, result in enumerate(results, 1):
            content = result['content']
            source = result.get('source', 'unknown')
            score = result.get('score', 0)
            
            # Clean up content (remove enhancement markers)
            content = re.sub(r'\[CS2 Topics:.*?\] ', '', content)
            content = re.sub(r'\[Strategy:.*?\] ', '', content)
            
            # Truncate if too long
            if len(content) > 400:
                content = content[:400] + "..."
            
            output.append(f"📄 **Document Section {i}** (Relevance: {score:.2f})")
            output.append(f"{content}\n")
        
        # Add context about information completeness
        if len(results) < 3:
            output.append("💡 *Limited information found in documents. Consider using web search for additional details.*")
        
        return "\n".join(output)

# Enhanced document tool
def create_enhanced_document_tool():
    """Create the enhanced document tool"""
    processor = EnhancedDocumentProcessor()
    
    def query_enhanced_documents(query: str) -> str:
        return processor.query_documents(query)
    
    return Tool(
        name="query_enhanced_documents",
        func=query_enhanced_documents,
        description="Search through enhanced CS2-specific documents using advanced vector embeddings. Provides comprehensive information about skins, trading strategies, market analysis, and pattern guides with improved semantic understanding."
    )

# Initialize the enhanced document tool
enhanced_document_tool = create_enhanced_document_tool() 