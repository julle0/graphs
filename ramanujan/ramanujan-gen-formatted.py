import random
from sympy.solvers.diophantine.diophantine import sum_of_squares
from sympy import isprime
from itertools import permutations, product
import numpy as np
import math
from scipy.sparse.linalg import eigsh
import sparse
# Based on the LPS construction of Ramanujan graphs
# find two primes p and q with conditions specified below:
def prime_finder(p_range):
    #init_rand = random.randint(p_range[0], p_range[1])
    init_rand = 1
    step = 2
    p = init_rand+1 if init_rand % 2 == 0 else init_rand
    while not (isprime(p) and p%4 == 1):
        p += step
    # Next, we will find the prime q:
    q_init = math.ceil(2*p**(1/2))
    q =  q_init+1 if q_init % 2 == 0 else q_init
    while not (isprime(q) and q%4 == 1 and p**((q-1)//2) % q  == 1):
        q += step
    return p, q

# We need an integer u that satisfies u**2 = -1 mod q
def find_u(q):
    for i in range(2,q-1):
        u = i**(int(q-1)/4) % q
        if u**2 % q == q-1:
            return u
    raise RuntimeError("No such u exists.")

def decompose_p(p):
    sol = []
    for _sol in sum_of_squares(p,4,zeros=True):
        _sol = list(_sol)
        _sol_map = [i%2 for i in _sol]
        odd_count = sum(_sol_map)
        if odd_count == 1:
            for i,k in enumerate(_sol_map):
                if k == 1:
                    _sol[0], _sol[i] = _sol[i], _sol[0]
            for perm in set(permutations(_sol[1:])):
                nonzero_ind = [i for i,k in enumerate(perm) if k!=0]
                for sign in product([1,-1], repeat=len(nonzero_ind)):
                    perm_list = list(perm)
                    for i, _sign in zip(nonzero_ind, sign):
                        perm_list[i] *= _sign
                    perm_list.insert(0, _sol[0])
                    sol.append(perm_list)
        else:
            continue
    if len(sol) == 0: raise RuntimeError("No suitable solutions exist.")
    else: return sol

# find the elements of S
def generate_S(solutions, u, p, q) -> np.array:
    # Find a normalisation constant to ensure that all determinants are 1 mod q
    S = []
    n = 1
    while np.sqrt(p % q + q*n) != int(np.sqrt(p % q + q*n)):  
    # look for an integer which is congruent to p mod q 
    # but also an integer when we take its square root
        n += 1
    norm_const = (int(np.sqrt(p % q + q*n))) % q
    inv_norm_const = 1
    while (norm_const*inv_norm_const) % q != 1:  
    # find the multiplicative inverse of this integer mod q
        inv_norm_const += 1
    # a,b,c,d come from the decomposition of p and u from find_u()
    for sol in solutions:
    # calculate the matrix entries which are of form
        v = np.zeros((2,2))  
        v[0,0] += (inv_norm_const*(sol[0] + u*sol[1])) % q  # a+ub 
        v[0,1] += (inv_norm_const*(sol[2] + u*sol[3])) % q  # c+ud 
        v[1,0] += (inv_norm_const*(-sol[2] + u*sol[3])) % q  
        v[1,1] += (inv_norm_const*(sol[0] - u*sol[1])) % q
        S.append(v)
    for v in S:
    # check that all matrices have det 1 mod q to a certain 
    # tolerance since numpy creates rounding errors
        det = np.linalg.det(v) % q
        if abs(det-1) > 0.001:
            raise ValueError(f'Something went wrong, det({v})%{q} = {det}, not 1')
    if len(S) != p+1:
        raise ValueError(f'Length of S != {p + 1} but {len(S)}')
    return np.array(S, dtype=np.int64)

# Generate G = PSL(2, F_q)
def generate_G(q: int) -> np.array:
    # elements of G need to have determinant = 1 
    # and an entry's inverse mod q may not be present,
    G = []
    # seen represents all of PSL(2, F_q) with det = 1
    seen = []  
    # or in other words, all of PSL(2, F_q) modulo +-the identity matrix
    for i in range(q**4):
     # PSL(2,F_q) contains all 2x2 matrices 
     # with entries in F_q, so q**4 unique matrices
        digits = []
     # turn base 10 integers in q**4 into base q representation
     # This loop code is based on a post on Stackexchange
        while i:  
            digits.append(int(i % q))
            i //= q
        if len(digits) < 4:
            for _ in range(4-len(digits)):
                digits.append(0)
        # enforce the det = 1 condition
        if (digits[0]*digits[3]-digits[1]*digits[2]) % q == 1:  
            seen.append(tuple(digits))  
     # reducing the list so, that each matrix and 
     # its inverse pair gets turned 
     # into a pair of either itself or its inverse
    for dts in seen: 
         G.append(min(dts, tuple((-x)%q for x in dts)))
    # deterministically pick either the original 
    # or the inverse, now governed by taking min() of two tuples
    # Remove duplicate elements
    G = list(set(G))
    if len(G) != q*(q**2-1)/2:
        raise ValueError(f'Length of G != {q*(q**2-1)/2} but {len(G)}')
    return np.array([np.reshape(np.array(G_tuple, dtype=np.int64), (2,2)) for G_tuple in G])
     # dedupe the list so we are left with 
     # mod +-identity. this final list 
     # should have q(q^2-1)/2 elements.

# Identify np.array with some bitstring/hash
def create_hash(input_M, q):
        # Canonicalization:
        M = np.ascontiguousarray(input_M)
        neg_M = np.ascontiguousarray((-input_M) % q)
        return min(M.tobytes(), neg_M.tobytes())
# Generate the adjacency matrix:
def generate_adj_matrix(g_map, S, p, q):
    # G is the label for vertices
    n = len(G)
    adj_matrix = np.zeros((n, n), dtype = np.uint8)
    # We need to hash the elements of g.
    for s in S:
        for i, g in enumerate(G):
            prod_hash = create_hash((s @ g) % q, q)
            j = g_map[prod_hash]
            adj_matrix[i,j] += 1
    col_sums = [col_sum for col_sum in adj_matrix.sum(0)]
    row_sums = [row_sum for row_sum in adj_matrix.sum(1)]

    if not all(c == p + 1 for c in col_sums):
        raise RuntimeError(f'Column weight not consistently p + 1 = {p+1}: {col_sums}')
    if not all(r == p + 1 for r in col_sums):
        raise RuntimeError(f'Row weight not consistently p + 1 = {p+1}: {row_sums}')
    
    return adj_matrix
# Generate the Cayley graph (G, S) with a G-lift:
def generate_edges(G,S,q):
    explicit_E = []
    # For every 0,g there exists len(S) amount of 1,sg
    for g in G:
        explicit_E.append([])
        for s in S:
            explicit_E[-1].append(((0,g), (1, (s @ g) % q)))
    return explicit_E

def check_ramanujan_bound(adj_matrix, p):
    eigvals = list(eigsh(adj_matrix, k=4, which='LM', return_eigenvectors=False))
    eigvals = [np.round(abs(x), decimals=5) for x in eigvals]
    eigvals = [x for x in eigvals if x != p + 1]
    max_eigval = max(eigvals)
    return True if max_eigval <= 2*np.sqrt(p) else False

def generate_adjanency(g_map, explicit_edges, adj, q):
    print(explicit_edges[1][1][1])
    # Associate some bit with some edge, and
    # parity check matrix for each vertex.
    w = sum(adj[0])
    r = w//8 # From notes
    # Random local parity check matrix
    h = np.random.randint(0, 2, size=(r,w))
    # For each vertex, the sum of the edges has to be 0
    # Firstly we need to form subsets of edges, per vertex
    # Data structure for Cayley: ((0,g), (1,sg))
    vertex_set = {g_id: [] for g_id in g_map.keys()}
    for i in explicit_edges:
        vertex_set[create_hash(explicit_edges[1][1][1], q)]
    print(vertex_set)

def lifted_product_shuffle(g_map, s_map, prod_map):
    r = len(s_map)
    # Generate E>, E^, F and V which belong to gamma.
    E_vertical, E_up, F, V = [], [], [], []
    v1, v2 = 0, 1
    # Shuffle shuffle
    for g in g_map:
        for s in s_map:
            E_vertical.append([s,g,v1])
            E_vertical.append([s,g,v2])

            E_up.append([v1, g, s])
            E_up.append([v2,g,s])

            for s_prime in s_map:
                #if s_prime == s:
                #    continue
                #else: F.append((s,g,s_prime))
                F.append((s,g,s_prime))

            V.append([v1,g,v2])
            V.append([v2,g,v1])

    r_range_list =[i for i in range(r)]
    perms = []

    for i in range(r):
        for k in range(r):
            perms.append([i,k])
    size_perms = len(perms)
    size_E_V = len(E_vertical)
    # E_vertical, E_up and V have the same size
    size_F = len(F)
    for k in range(size_E_V):
        r_mod = k % r
        perms_mod = k % size_perms
        E_vertical[k].append(r_range_list[r_mod])
        E_up[k].append(r_range_list[r_mod])
        V[k].append(perms[perms_mod])

    return E_vertical, E_up, F, V
# TARKISTA ONKO OIKEIN LMAO
def calc_stabilizer(h, E_vertical, E_up, F, V):
    r = len(s_map)
    print("h: ",h,"\nh_prime: ", h_prime)
    # Our Matrix M has the dimensions len(F), len(E_vertical)
    print(len(F))
    print(len(E_vertical))
    def check_incidence(a, b):
        for i,x in enumerate(a):
            if x == b[i]
                return True
        return False
    # Cannot do calculations with M, N (out of RAM)
    M = np.zeros(shape=(len(F), len(E_vertical)), dtype=np.uint8)
    N = np.zeros(shape=(len(E_vertical, len(V))), dtype=np.uint8)
    # Generate M:
    for i, e in enumerate(E_vertical):
        j = 0
        incidence = False
        for j,f in enumerate(F):
            if check_incidence(f,e):
                incidence = True
                break
            else:
                continue
        if h[i,j] == 1 and incidence:
            M[i,j] += 1
    # Generate N:
    for i,v in enumerate(V)
        j = 0
        incidence = False
        for j, e in enumerate(E_vertical):
            if check_incidence(e,v)
                incidence = True
                break
            else:
                continue
        if (h[i,j] == 1) and incidence and (j=v[-1]):
            N[i,j] += 1
    return np.array([np.transpose(M), N])

if __name__ == "__main__":
    primes = prime_finder([0, 50])
    p = primes[0]
    q = primes[1]
    u = find_u(q)
    print(f"p,q,u: {p}, {q}, {u}")
    decomposition = decompose_p(p)

    S = generate_S(decomposition, u, p, q)
    G = generate_G(q)
    print(len(G), len(S))
    s_map = {create_hash(s,q): i for i, s in enumerate(S)}
    g_map = {create_hash(g,q): i for i, g in enumerate(G)}

    adj = generate_adj_matrix(g_map, S, p, q)
    print(
f'''Ramanujan (G,S) created, |G|={len(G)}, |S|={len(S)}
with adjacency dim={adj.size}
''')
    #IS_RAMANUJAN = check_ramanujan_bound(adj,p)
    #print(IS_RAMANUJAN)
    # We have now succeeded in creating a Ramanujan graph
    # We shall now create the qLDPC of 2 distinct Tanner graphs
    # And combine these 2 with a lifted-product.
    explicit_edges = generate_edges(G, S, q)
    E_vertical, E_up, F, V = lifted_product_shuffle(g_map, s_map, prod_map)
    h = np.random.randint(0,2, size=(r//8, r))
    h_prime = np.random.randint(0, 2, size=(r//8, r))
    dual_H_Z = calc_stabilizer(h, E_vertical, E_up, F, V, g_map, s_map, G, S)
    H_X = calc_stabilizer(h_prime, E_vertical, E_up, F, V, g_map, s_map, G, S)
