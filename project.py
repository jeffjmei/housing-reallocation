import numpy

def floyd_warshall(distance_matrix):
    """
    DESCRIPTION: finds the shortest distance between every node
    """
    n = len(distance_matrix)
    dist = np.array(distance_matrix, dtype=float)
    next_node = [
        [None if dist[i][j] == float('inf') else j 
        for j in range(n)] for i in range(n)
    ]

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
            #if i == j:
            #    D[i][j] = 0  

    return D

def cycle_diagonals(D_prime, A):
    """
    Adjusts the diagonal entries of D_prime to reflect shortest cycle lengths,
    by checking only arcs (i → j) that actually exist in A.
    """
    n = len(D_prime)
    D_cycle = D_prime.copy()
    for i in range(n):
        min_cycle_len = float('inf')
        for j in range(n):
            if i != j and A[i][j] > 0 and D_prime[j][i] < float('inf'):
                cycle_len = 1 + D_prime[j][i]
                if cycle_len < min_cycle_len:
                    min_cycle_len = cycle_len
        D_cycle[i][i] = min_cycle_len
    return D_cycle

# Identify Circuits
def find_paths_of_length_k(A, D_prime, start, end, k):
    """
    Find the first path from `start` to `end` of exact length k using arcs where A[i][j] > 0.
    Stops at the first valid path.

    Returns:
        A list of node indices representing the path, or None if not found.
    """
    n = len(A)
    result = []

    def dfs(path, visited, length_so_far):
        nonlocal result
        if result:  # early exit if path already found
            return

        u = path[-1]
        if u == end and length_so_far == k:
            result = path[:]
            return

        for v in range(n):
            if A[u][v] > 0 and (v not in visited or v == end):
                path.append(v)
                visited.add(v)
                dfs(path, visited, length_so_far + 1)
                path.pop()
                visited.discard(v)

    dfs([start], {start}, 0)
    return result if result else None

def find_all_paths_of_length_k(A, D_prime, start, end, k):
    """
    Find all paths from `start` to `end` of exact length k using arcs where A[i][j] > 0.

    Returns:
        A list of paths, where each path is a list of node indices.
    """
    n = len(A)
    results = []

    def dfs(path, visited, length_so_far):
        u = path[-1]

        if u == end and length_so_far == k:
            results.append(path[:])
            return

        if length_so_far >= k:
            return  # avoid overstepping

        for v in range(n):
            if A[u][v] > 0 and (v not in visited or v == end):
                path.append(v)
                visited.add(v)
                dfs(path, visited, length_so_far + 1)
                path.pop()
                visited.discard(v)

    dfs([start], {start}, 0)
    return results

def masked_distance_matrix(D_prime, s):
    """
    Returns a modified copy of D_prime where only row s, column s,
    and entries equal to 1 are retained. All other entries set to ∞.
    """
    n = len(D_prime)
    D_masked = np.full((n, n), float('inf'))

    for i in range(n):
        for j in range(n):
            if i == s or j == s or D_prime[i][j] == 1:
                D_masked[i][j] = D_prime[i][j]

    return D_masked

def get_bottleneck_demand(path, A):
    """
    Given a path as a list of node indices and matrix A,
    return the minimum A[i][j] along the path.
    Assumes path is a cycle or path: [i0, i1, ..., in]
    so arcs are (i0→i1), (i1→i2), ..., (in-1→in)
    """
    return min(A[i][j] for i, j in zip(path[:-1], path[1:]))

def adjust_circuit_flow(A, path, demand):
    """
    Subtracts `bottleneck` from each arc along the given path in A.

    Parameters:
    - A: numpy 2D array (updated in-place)
    - path: list of node indices, e.g., [0, 1, 4, 0]
    - bottleneck: scalar to subtract along each arc
    """
    for i, j in zip(path[:-1], path[1:]):
        A[i][j] -= demand
        if A[i][j] == 0:
             D_prime[i][j] = float('inf')


# Run Program
A = [
    [0, 6, 0, 7, 0, 3, 0, 6, 5, 0],
    [0, 0, 0, 3, 17, 12, 0, 0, 9, 7],
    [8, 4, 0, 0, 0, 8, 10, 0, 0, 5],
    [0, 0, 8, 0, 3, 0, 7, 0, 0, 0],
    [6, 0, 4, 0, 0, 10, 6, 5, 10, 0],
    [0, 22, 0, 8, 0, 0, 3, 4, 6, 0],
    [6, 7, 13, 0, 0, 0, 0, 0, 8, 0],
    [0, 9, 5, 0, 0, 6, 0, 11, 6, 0],
    [0, 0, 5, 0, 16, 0, 19, 0, 0, 4],
    [7, 0, 0, 5, 4, 0, 0, 0, 0, 0]
]

# Derive Initial Distance Matrix
D = initial_distance_matrix(A)
D_prime, _ = floyd_warshall(D)
D_prime = cycle_diagonals(D_prime, A)

# Find Shortest path
s = 0
path_len = int(D_prime[s][s])
D_prime_masked = masked_distance_matrix(D_prime, s)
for t in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    #path = find_paths_of_length_k(A, D_prime_masked, t, s, path_len - 1)
    paths = find_all_paths_of_length_k(A, D_prime_masked, t, s, path_len - 1)
    for path in paths: 
        if path is None:
            continue         
        else: 
            path = [s] + path

        # find bottleneck demand and adjust flow
        demand = get_bottleneck_demand(path, A)
        adjust_circuit_flow(A, path, demand)
        if demand > 0: 
            print(f"Path: {path}; {demand} Sets")

