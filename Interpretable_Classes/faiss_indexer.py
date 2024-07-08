import faiss
import numpy as np

class FaissIndexer:
    def __init__(self, feature_dim):
        """
        Initialize the Faiss indexer with the given feature dimension.
        
        Parameters:
        feature_dim (int): The dimension of the feature vectors to be indexed.
        """
        self.index = faiss.IndexFlatL2(feature_dim)  # Initialize a FAISS index for L2 distance.

    def add_features(self, features):
        """
        Add feature vectors to the FAISS index.
        
        Parameters:
        features (np.ndarray): An array of feature vectors to be added to the index.
        """
        self.index.add(features)

    def search_k_nearest_neighbors(self, query_feature, k=5):
        """
        Perform k-nearest neighbor search on the FAISS index.
        
        Parameters:
        query_feature (np.ndarray): The query feature vector.
        k (int): The number of nearest neighbors to retrieve.
        
        Returns:
        distances (np.ndarray): Distances to the nearest neighbors.
        indices (np.ndarray): Indices of the nearest neighbors.
        """
        distances, indices = self.index.search(query_feature, k)
        return distances, indices

    def save(self, path):
        """
        Save the FAISS index to a file.
        
        Parameters:
        path (str): The path where the index will be saved.
        """
        faiss.write_index(self.index, path)

    @staticmethod
    def load(path):
        """
        Load a FAISS index from a file.
        
        Parameters:
        path (str): The path from where the index will be loaded.
        
        Returns:
        FaissIndexer: An instance of FaissIndexer with the loaded index.
        """
        index = faiss.read_index(path)
        return FaissIndexer._from_index(index)

    @staticmethod
    def _from_index(index):
        """
        Create a FaissIndexer instance from an existing FAISS index.
        
        Parameters:
        index (faiss.IndexFlatL2): An existing FAISS index.
        
        Returns:
        FaissIndexer: An instance of FaissIndexer with the provided index.
        """
        instance = FaissIndexer(index.d)  # Initialize with the feature dimension.
        instance.index = index  # Set the index to the provided FAISS index.
        return instance
