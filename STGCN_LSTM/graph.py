import torch
import numpy as np

class Graph:
    def __init__(self, num_node):
        self.num_node = num_node
        self.AD, self.AD2, self.AD3 = self.normalize_adjacency()
        
    def normalize_adjacency(self):
        # Define self-connections
        self_link = [(i, i) for i in range(self.num_node)]
        
        # Define base neighbor connections (1-based indexing in original data)
        neighbor_1base = [(1, 5), (1, 2), (2, 3), (3, 4), (4, 8), (5, 6), (6, 7), 
                         (7, 9), (11, 10), (12, 13), (12, 14), (14, 16), (16, 22), 
                         (16, 18), (16, 20), (20, 18), (13, 15), (15, 17), (17, 23), 
                         (17, 19), (19, 21), (17, 21), (13, 25), (25, 27), (27, 29), 
                         (29, 33), (29, 31), (31, 33), (12, 24), (24, 26), (26, 28), 
                         (28, 30), (28, 32), (30, 32)]
        
        # Convert to 0-based indexing
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link
        
        # Create first order adjacency matrix
        A = np.zeros((self.num_node, self.num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1
        
        # Create second order adjacency matrix
        A2 = np.zeros((self.num_node, self.num_node))
        for root in range(A.shape[1]):
            for neighbour in range(A.shape[0]):
                if A[root, neighbour] == 1:
                    for neighbour_of_neigbour in range(A.shape[0]):
                        if A[neighbour, neighbour_of_neigbour] == 1:
                            A2[root, neighbour_of_neigbour] = 1
        
        # Create third order adjacency matrix
        A3 = np.zeros((self.num_node, self.num_node))
        for root in range(A2.shape[1]):
            for neighbour in range(A2.shape[0]):
                if A2[root, neighbour] == 1:
                    for neighbour_of_neigbour in range(A2.shape[0]):
                        if A2[neighbour, neighbour_of_neigbour] == 1:
                            A3[root, neighbour_of_neigbour] = 1
        
        # Normalize all matrices
        AD = self.normalize(A)
        AD2 = self.normalize(A2)
        AD3 = self.normalize(A3)
        
        return AD, AD2, AD3
    
    def normalize(self, adjacency):
        rowsum = np.array(adjacency.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = np.diag(r_inv)
        normalize_adj = r_mat_inv.dot(adjacency)
        normalize_adj = normalize_adj.astype('float32')
        normalize_adj = torch.FloatTensor(normalize_adj)
        return normalize_adj
    
    def to(self, device):
        """
        Move the graph adjacency matrices to specified device (CPU/GPU)
        """
        self.AD = self.AD.to(device)
        self.AD2 = self.AD2.to(device)
        self.AD3 = self.AD3.to(device)
        return self