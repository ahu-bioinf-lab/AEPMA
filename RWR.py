import numpy as np

def generate_node_embeddings(adj_matrix, num_steps = 80, num_restarts = 10, restart_prob = 0.2, embedding_dim = 64):
    n_nodes = adj_matrix.shape[0]
    P = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        degree = np.sum(adj_matrix[i, :])
        if degree > 0:
            P[i, :] = adj_matrix[i, :] / degree

    # Perform restart random walks and accumulate transition probabilities
    A = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(num_restarts):
            curr_node = i
            for k in range(num_steps):
                A[curr_node, i] += 1
                if np.random.rand() < restart_prob:
                    curr_node = i
                else:
                    next_node = np.random.choice(n_nodes, p=P[curr_node, :])
                    curr_node = next_node

    # Compute node embeddings using singular value decomposition
    _, s, V = np.linalg.svd(A, full_matrices=False)
    embeddings = np.dot(V[:embedding_dim, :].T, np.diag(s[:embedding_dim]))

    return embeddings
