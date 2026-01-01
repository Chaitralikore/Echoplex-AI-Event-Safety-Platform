"""
Vector Database Module
Scalable vector storage and search with metadata filtering

Supports:
- ChromaDB (lightweight, embedded - default)
- Milvus (production scale, if available)
- In-memory fallback (development)

Features:
- 512-dimensional vector storage
- Metadata filtering (color, gender, age range, etc.)
- Approximate Nearest Neighbor (ANN) search
- Scalable to millions of detections
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import os

# Try to import vector database libraries
CHROMADB_AVAILABLE = False
MILVUS_AVAILABLE = False

try:
    import chromadb
    CHROMADB_AVAILABLE = True
    print("ChromaDB available for vector search")
except ImportError:
    pass

try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
    MILVUS_AVAILABLE = True
    print("Milvus available for production-scale vector search")
except ImportError:
    pass


class VectorDatabase:
    """
    Scalable vector database for person re-identification
    
    Schema:
    - id: Unique detection ID
    - vector: 512-d feature vector
    - metadata:
        - upper_color: Primary upper body color
        - lower_color: Primary lower body color
        - gender: Detected or reported gender
        - timestamp: Detection timestamp
        - source: Camera/video ID
        - description: Free-text description
    """
    
    def __init__(self, db_path: str = "./vector_db", collection_name: str = "person_detections"):
        """
        Initialize vector database
        
        Args:
            db_path: Path for persistent storage
            collection_name: Name of the collection/index
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.vector_dim = 512
        
        self.client = None
        self.collection = None
        self.backend = "memory"  # Track which backend is in use
        
        # Initialize the best available backend
        if CHROMADB_AVAILABLE:
            self._init_chromadb()
        elif MILVUS_AVAILABLE:
            self._init_milvus()
        else:
            self._init_memory()
    
    def _init_chromadb(self):
        """Initialize ChromaDB (embedded, lightweight)"""
        try:
            os.makedirs(self.db_path, exist_ok=True)
            
            # Use modern PersistentClient API (ChromaDB >= 0.4.0)
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            self.backend = "chromadb"
            print(f"ChromaDB initialized at {self.db_path}")
            
        except Exception as e:
            print(f"ChromaDB initialization failed: {e}")
            self._init_memory()
    
    def _init_milvus(self):
        """Initialize Milvus (production scale)"""
        try:
            connections.connect("default", host="localhost", port="19530")
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
                FieldSchema(name="upper_color", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="lower_color", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="gender", dtype=DataType.VARCHAR, max_length=20),
                FieldSchema(name="timestamp", dtype=DataType.INT64),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=100),
            ]
            
            schema = CollectionSchema(fields, description="Person Re-ID detections")
            
            # Create or get collection
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
            else:
                self.collection = Collection(self.collection_name, schema)
                
                # Create IVF_FLAT index for fast search
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
                self.collection.create_index("vector", index_params)
            
            self.collection.load()
            self.backend = "milvus"
            print("Milvus initialized for production-scale search")
            
        except Exception as e:
            print(f"Milvus initialization failed: {e}")
            self._init_memory()
    
    def _init_memory(self):
        """Initialize in-memory storage (fallback)"""
        self.vectors = []  # List of (id, vector, metadata) tuples
        self.backend = "memory"
        print("Using in-memory vector storage (development mode)")
    
    def add_detection(self, detection_id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """
        Add a person detection to the database
        
        Args:
            detection_id: Unique ID for this detection
            vector: 512-d feature vector
            metadata: Dict containing:
                - upper_color: str
                - lower_color: str
                - gender: str (optional)
                - timestamp: int (unix timestamp)
                - source: str (camera/video ID)
                - description: str (optional)
        
        Returns:
            Success status
        """
        try:
            vector = vector.astype(np.float32).tolist()
            
            if self.backend == "chromadb":
                self.collection.add(
                    ids=[detection_id],
                    embeddings=[vector],
                    metadatas=[metadata]
                )
            
            elif self.backend == "milvus":
                self.collection.insert([
                    [detection_id],
                    [vector],
                    [metadata.get('upper_color', 'unknown')],
                    [metadata.get('lower_color', 'unknown')],
                    [metadata.get('gender', 'unknown')],
                    [int(metadata.get('timestamp', 0))],
                    [metadata.get('source', 'unknown')]
                ])
            
            else:  # memory
                self.vectors.append({
                    'id': detection_id,
                    'vector': np.array(vector),
                    'metadata': metadata
                })
            
            return True
            
        except Exception as e:
            print(f"Error adding detection: {e}")
            return False
    
    def search(self, 
               query_vector: np.ndarray, 
               top_k: int = 10,
               metadata_filter: Optional[Dict[str, Any]] = None,
               min_similarity: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search for similar detections with optional metadata filtering
        
        Args:
            query_vector: 512-d feature vector to search for
            top_k: Number of results to return
            metadata_filter: Dict of metadata constraints, e.g.:
                {
                    "upper_color": "red",      # Exact match
                    "lower_color": ["blue", "black"],  # Any of these
                    "gender": "female"
                }
            min_similarity: Minimum cosine similarity threshold
            
        Returns:
            List of matches with id, similarity, and metadata
        """
        query_vector = query_vector.astype(np.float32)
        
        try:
            if self.backend == "chromadb":
                return self._search_chromadb(query_vector, top_k, metadata_filter, min_similarity)
            
            elif self.backend == "milvus":
                return self._search_milvus(query_vector, top_k, metadata_filter, min_similarity)
            
            else:  # memory
                return self._search_memory(query_vector, top_k, metadata_filter, min_similarity)
                
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _search_chromadb(self, query_vector: np.ndarray, top_k: int,
                         metadata_filter: Optional[Dict], min_similarity: float) -> List[Dict]:
        """Search using ChromaDB"""
        where_clause = None
        
        if metadata_filter:
            where_clauses = []
            for key, value in metadata_filter.items():
                if isinstance(value, list):
                    where_clauses.append({key: {"$in": value}})
                else:
                    where_clauses.append({key: value})
            
            if len(where_clauses) == 1:
                where_clause = where_clauses[0]
            elif len(where_clauses) > 1:
                where_clause = {"$and": where_clauses}
        
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            where=where_clause,
            include=["embeddings", "metadatas", "distances"]
        )
        
        matches = []
        if results['ids'] and len(results['ids']) > 0:
            for i, id_ in enumerate(results['ids'][0]):
                # ChromaDB returns distance, convert to similarity
                distance = results['distances'][0][i] if results['distances'] else 0
                similarity = 1 - distance  # For cosine distance
                
                if similarity >= min_similarity:
                    matches.append({
                        'id': id_,
                        'similarity': similarity,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                    })
        
        return matches
    
    def _search_milvus(self, query_vector: np.ndarray, top_k: int,
                       metadata_filter: Optional[Dict], min_similarity: float) -> List[Dict]:
        """Search using Milvus"""
        # Build expression for filtering
        expr = ""
        if metadata_filter:
            conditions = []
            for key, value in metadata_filter.items():
                if key in ['upper_color', 'lower_color', 'gender', 'source']:
                    if isinstance(value, list):
                        values_str = ', '.join([f'"{v}"' for v in value])
                        conditions.append(f'{key} in [{values_str}]')
                    else:
                        conditions.append(f'{key} == "{value}"')
            
            if conditions:
                expr = " && ".join(conditions)
        
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr if expr else None,
            output_fields=["upper_color", "lower_color", "gender", "timestamp", "source"]
        )
        
        matches = []
        for hits in results:
            for hit in hits:
                similarity = 1 - hit.distance  # Convert distance to similarity
                if similarity >= min_similarity:
                    matches.append({
                        'id': hit.id,
                        'similarity': similarity,
                        'metadata': {
                            'upper_color': hit.entity.get('upper_color'),
                            'lower_color': hit.entity.get('lower_color'),
                            'gender': hit.entity.get('gender'),
                            'timestamp': hit.entity.get('timestamp'),
                            'source': hit.entity.get('source')
                        }
                    })
        
        return matches
    
    def _search_memory(self, query_vector: np.ndarray, top_k: int,
                       metadata_filter: Optional[Dict], min_similarity: float) -> List[Dict]:
        """Search in-memory storage (brute force)"""
        candidates = []
        
        for item in self.vectors:
            # Apply metadata filter
            if metadata_filter:
                match = True
                for key, value in metadata_filter.items():
                    item_value = item['metadata'].get(key)
                    if isinstance(value, list):
                        if item_value not in value:
                            match = False
                            break
                    else:
                        if item_value != value:
                            match = False
                            break
                
                if not match:
                    continue
            
            # Compute similarity
            similarity = float(np.dot(query_vector, item['vector']))
            
            if similarity >= min_similarity:
                candidates.append({
                    'id': item['id'],
                    'similarity': similarity,
                    'metadata': item['metadata']
                })
        
        # Sort by similarity and take top_k
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        return candidates[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if self.backend == "chromadb":
            count = self.collection.count()
        elif self.backend == "milvus":
            count = self.collection.num_entities
        else:
            count = len(self.vectors)
        
        return {
            'backend': self.backend,
            'total_detections': count,
            'vector_dimension': self.vector_dim
        }
    
    def clear(self):
        """Clear all detections from the database"""
        if self.backend == "chromadb":
            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        elif self.backend == "milvus":
            self.collection.drop()
        else:
            self.vectors = []


# Global instance
vector_db = VectorDatabase()
