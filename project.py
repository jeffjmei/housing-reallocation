import numpy

def floyd_warshall(distance_matrix):
    """
    DESCRIPTION: finds the shortest distance between every node
    """
    n = len(distance_matrix)
    dist = np.array(distance_matrix, dtype=float)
    next_node = [[None if dist[i][j] == float('inf') else j for j in range(n)] for i in range(n)]

    for k in range(n): # node k is intermediate node
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]  # Update path 

    return dist, next_node

def reconstruct_path(u, v, next_node):
    """
    DESCRIPTION: reconstructs path from floyd-warshall output
    """
    if next_node[u][v] is None:
        return []

    path = [u]
    while u != v:
        u = next_node[u][v]
        if u is None:
            return []  # No path
        path.append(u)
    return path

def initial_distance_matrix(A):
    """
    DESCRIPTION: finds shortest number of moves between 
        housing categories. 
    """
    n = len(A)
    D = np.full((n, n), float('inf'))

    for i in range(n):
        for j in range(n):
            # non-empty
            if A[i][j] > 0 and A[i][j] < float('inf'):
                D[i][j] = 1
            # diagonal
            if i == j:
                D[i][j] = 0  

    return D

def cycle_diagonals(D_prime):
    """
    Adjusts the diagonal entries of D_prime to represent shortest cycle lengths.
    This function should be run AFTER Floyd-Warshall.

    Parameters:
    D_prime: np.array (n x n), result of Floyd-Warshall
    
    Returns:
    D_cycle: np.array (n x n), with adjusted diagonals
    """
    n = len(D_prime)
    D_cycle = D_prime.copy()
    for i in range(n):
        min_cycle_len = float('inf')
        for j in range(n):
            if i != j and D_prime[i][j] < float('inf') and D_prime[j][i] < float('inf'):
                cycle_len = D_prime[i][j] + D_prime[j][i]
                if cycle_len < min_cycle_len:
                    min_cycle_len = cycle_len
        D_cycle[i][i] = min_cycle_len
    return D_cycle

def find_all_cycles_of_length_k(s, A, k, next_node):
    """
    Find all simple cycles of length k starting and ending at s,
    using only arcs with A[i][j] > 0.

    Parameters:
    - s: starting node
    - A: 2D numpy array
    - k: target cycle length
    - next_node: Floyd-Warshall next-hop matrix

    Returns:
    - List of cycles (each a list of nodes starting and ending at s)
    """
    n = len(A)
    paths = []

    def dfs(path, visited):
        current = path[-1]

        # Termination
        if len(path) == k:
            if A[current][s] > 0:
                paths.append(path + [s])
            return

        for neighbor in range(n):
            if A[current][neighbor] > 0 and neighbor not in visited:
                dfs(path + [neighbor], visited | {neighbor})

    dfs([s], {s})
    return paths

def reallocate_from_node(A, s, max_cycle_length=None):
    """
    Perform Wright-style cycle-based reallocation from node s:
    - Start from D'[s,s]
    - At each length k, find all cycles of length k from s
    - Apply max flow through all such cycles
    - Repeat until no more cycles can be found

    Parameters:
    - A: 2D numpy array (will be modified in-place)
    - s: int, starting node
    - max_cycle_length: optional int, upper limit on cycle length

    Returns:
    - List of (circuit, flow) tuples executed
    """
    n = len(A)
    executed = []

    while True:
        # Rebuild D and D' each round
        D = initial_distance_matrix(A)
        D_prime, next_node = floyd_warshall(D)
        D_prime = cycle_diagonals(D_prime)

        cycle_len = int(D_prime[s][s])
        if cycle_len == float('inf'):
            break

        if max_cycle_length is not None and cycle_len > max_cycle_length:
            break

        # Find all cycles of length cycle_len from s
        cycles = find_all_cycles_of_length_k(s, A, cycle_len, next_node)
        if not cycles:
            break  # No more cycles of this length

        for circuit in cycles:
            edges = [(circuit[i], circuit[i + 1]) for i in range(len(circuit) - 1)]
            m = min(A[i][j] for i, j in edges)
            if m == 0:
                continue
            for i, j in edges:
                A[i][j] -= m
            executed.append((circuit, m))

    return executed

all_executed = []
for s in range(len(A)):
    executed = reallocate_from_node(A, s)
    all_executed.extend(executed)

for path, flow in all_executed:
    print(f"Moved {flow} families through: {path}")


# Run Program
A = [
    [float('inf'), 6, float('inf'), 7, float('inf'), 3, float('inf'), 6, 5, 0],
    [float('inf'), float('inf'), float('inf'), 3, 17, 12, float('inf'), float('inf'), 9, 7],
    [8, 4, float('inf'), float('inf'), float('inf'), 8, 10, float('inf'), float('inf'), 5],
    [float('inf'), float('inf'), 8, float('inf'), 3, float('inf'), 7, float('inf'), float('inf'), 0],
    [6, float('inf'), 4, float('inf'), float('inf'), 10, 6, 5, 10, 0],
    [float('inf'), 22, float('inf'), 8, float('inf'), float('inf'), 3, 4, 6, 0],
    [6, 7, 13, float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), 8, 0],
    [float('inf'), 9, 5, float('inf'), float('inf'), 6, float('inf'), 11, 6, 0],
    [float('inf'), float('inf'), 5, float('inf'), 16, float('inf'), 19, float('inf'), float('inf'), 4],
    [7, float('inf'), float('inf'), 5, 4, float('inf'), float('inf'), float('inf'), float('inf'), 0]
]

# Derive Initial Distance Matrix
D = initial_distance_matrix(A)
D_prime, _ = floyd_warshall(D)
D_prime = cycle_diagonals(D_prime)

# Identify Circuits
D_prime

def identify_circuit(D_prime, node):
    min_dist = D_prime[node,node]
    # find adjacent node
    adjacent_nodes = D_prime


