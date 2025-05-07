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

    return D

def cycle_diagonals(D_prime, A):
    """
    Adjusts the diagonal entries of D_prime to reflect shortest cycle lengths,
    by checking only arcs (i - j) that actually exist in A.
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

def find_all_paths_of_length_k(A, D_prime, start, end, k):
    """
    Finds all k-length circuits
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
    Used to hide D_prime so only the sth row and col can be seen
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
    Finds the maximum number of tenants we can shuffle on an arc
    """
    return min(A[i][j] for i, j in zip(path[:-1], path[1:]))

def adjust_circuit_flow(A, path, demand):
    """
    Shuffles tenants and readjusts demand 
    """
    for i, j in zip(path[:-1], path[1:]):
        A[i][j] -= demand
        if A[i][j] == 0:
             D_prime[i][j] = float('inf')

def extract_circuits_length_k(A, D_prime_masked, s, k):
    for t in range(len(A)):
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

D = initial_distance_matrix(A)
D_prime, _ = floyd_warshall(D)
D_prime = cycle_diagonals(D_prime, A)
while np.any((diag != 0) & (np.diag(D_prime) != float('inf'))): 
    print(ct)

    # Find Shortest path
    diag = np.diag(D_prime)
    mask = (diag != 0) & (diag != float('inf'))  
    s = int(np.argmax(mask))
    path_len = int(D_prime[s][s])
    D_prime_masked = masked_distance_matrix(D_prime, s)
    extract_circuits_length_k(A, D_prime_masked, s, path_len - 1)

    # Update Distance Matrix
    D = initial_distance_matrix(A)
    D_prime, _ = floyd_warshall(D)
    D_prime = cycle_diagonals(D_prime, A)


