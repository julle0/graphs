from sympy.solvers.diophantine.diophantine import sum_of_squares
from sympy import isprime
from itertools import permutations, product
import numpy as np
import math
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix, hstack
RAMANUJAN_CHECK = False
R_DIVISOR = 8
# Based on the LPS construction of Ramanujan graphs
# find two primes p and q with conditions specified below:
def prime_finder(p_range):
    init_rand = random.randint(p_range[0], p_range[1])
    step = 2
    p = init_rand+1 if init_rand % 2 == 0 else init_rand
    while not (isprime(p) and p%4 == 1 and (p+1)//R_DIVISOR!=0):
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

# We need to find a decomposition of p
# such that a**2+b**2+c**2+d**2=p, a>0, and b,c,d even.
def decompose_p(p):
    sol = []
    # sympy library's sum_of_squares gives
    # the solutions we need (non-permuated/cleaned-up)

    for _sol in sum_of_squares(p,4,zeros=True):
        _sol = list(_sol)
        _sol_map = [i%2 for i in _sol]
        odd_count = sum(_sol_map)
    # We need 3 even sols, 1 odd.
        if odd_count == 1:
    # Swap places with odd sol being index 0
            for i,k in enumerate(_sol_map):
                if k == 1:
                    _sol[0], _sol[i] = _sol[i], _sol[0]
        # This portion generates all permutations of our
        # even solutions, with all neg/pos possibilities.
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
            # Somehow invalid solution arises.
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

def check_ramanujan_bound(adj_matrix, p):
    eigvals = list(eigsh(adj_matrix, k=4, which='LM', return_eigenvectors=False))
    eigvals = [np.round(abs(x), decimals=5) for x in eigvals]
    eigvals = [x for x in eigvals if x != p + 1]
    max_eigval = max(eigvals)
    return True if max_eigval <= 2*np.sqrt(p) else False

def lifted_product_shuffle(g_map, s_map):
    r = len(s_map)//R_DIVISOR
    # Generate E>, E^, F and V which belong to gamma.
    E_horizontal, E_up, F, V = [], [], [], []
    v1, v2 = 0, 1
    # Shuffle shuffle
    for g in g_map.keys():
        for s in s_map.keys():
            E_horizontal.append([s,g,v1])
            E_horizontal.append([s,g,v2])

            E_up.append([v1,g,s])
            E_up.append([v2,g,s])

            for s_prime in s_map:
                #if s_prime == s:
                #    continue
                #else: F.append((s,g,s_prime))
                F.append((s,g,s_prime))

        V.append([v1,g,v2])
        V.append([v2,g,v1])

    r_range_list = [i for i in range(r)]
    # Augment our elements to their index basis.
    # E_horizontal x r, E_up x r, V x r x r''
    E_horizontal_aug = []
    E_up_aug = []
    V_aug = []
    for e in E_horizontal:
        for i in range(r):
            E_horizontal_aug.append([
                e[0], e[1], e[2], i])

    for e in E_up:
        for i in range(r):
            E_up_aug.append([
                e[0], e[1], e[2], i])

    for v in V:
        for i in range(r):
            for j in range(r):
                V_aug.append([
                    v[0], v[1], v[2], i,j])

    return E_horizontal_aug, E_up_aug, F, V_aug

def gen_stabilizer_matrix(h: np.array, E_horizontal: list, E_up: list, F: list, V: list, prod_map: dict) -> tuple:
    r = h.shape[0]
    rp = h.shape[1]
    if len(h) == 0:
        raise ValueError(f'Invalid h: {h}')
    # Based on notes
    def f_check_incidence(f, e):
        # each f is incident to (0,g,s')
        # and (1, sg, s')
        #print(f,e)
        s = f[0]
        g = f[1]
        s_prime = f[2]
        if e[2] != s_prime:
            return False
        elif e[0] == 0:
            return e[1] == g
        elif e[0] == 1:
            return e[1] == prod_map[(s, g)]
        else:
            return False

    def v_check_incidence(v,e):
        s = e[0]
        g = e[1]
        v_e = e[2]
        if v[2] != v_e:
            return False
        elif v[0] == 0:
            return v[1] == g
        elif v[0] == 1:
            return v[1] == prod_map[(s, g)]
        else: return False

    # The sizes of M,N are really big.
    # We thus compress a bit, using coo_matrix
    # (check create_stabilizer_matrix func)
    M_rows = []
    M_cols = []
    M_data = []
    N_rows = []
    N_cols = []
    N_data = []
    #M = np.zeros(shape=(len(F), len(E_horizontal)), dtype=np.uint8)
    #N = np.zeros(shape=(len(E_horizontal, len(V))), dtype=np.uint8)
    # Generate M:
    for edge_id, e in enumerate(E_up):
        i = e[-1]
        j = 0
        for face_id, f in enumerate(F):
            if f_check_incidence(f,e):
                if j>=rp:
                    # Just incase
                    break
                if h[i,j] == 1:
    # (e->,i) has to be row-major flattened (2d->1d mapping)
                    M_rows.append(edge_id*r+i)
                    M_cols.append(face_id)
                    M_data.append(1)
                j += 1
    print("M_data:", len(M_data))
    # Generate N:
    for vertex_id, v in enumerate(V):
        j = 0
        # V basis is (v..., a, b) -> b=k
        k = v[-1]
        i = v[-2]
        for edge_id, e in enumerate(E_horizontal):
        # E-> basis is (e, c) -> should be c=l
            l = e[-1]
            if v_check_incidence(v,e):
                if j>=rp:
                    break

                if h[i,j] == 1 and k == l:
                    # 3d -> 1d mapping
                    N_rows.append(vertex_id*(r*rp)+i*rp+k)
                    N_cols.append(edge_id*rp+l)
                    N_data.append(1)
                j += 1
    print("N_data:", len(N_data))

    return (M_rows, M_cols, M_data, N_rows, N_cols, N_data)

def create_stabilizer_matrix(M_rows, M_cols, M_data, N_rows, N_cols, N_data):
    M_sparse = coo_matrix((M_data, (M_rows, M_cols)),
                      shape=(len(E_horizontal) * r, len(F)))
    N_sparse = coo_matrix((N_data, (N_rows, N_cols)),
                      shape=(len(V) * r, len(E_vertical)))
    stabilizer = hstack([M_sparse.T, N_sparse])
    return stabilizer
if __name__ == "__main__":
    search_range_max =  int(input("Max value for semi-random p: "))
    primes = prime_finder([0, search_range_max])
    p = primes[0]
    q = primes[1]
    u = find_u(q)
    print(f"Intial values p,q,u: {p}, {q}, {u}")
    decomposition = decompose_p(p)
    print("Generating (G,S) pair, and their mappings...")
    S = generate_S(decomposition, u, p, q)
    G = generate_G(q)
    s_map = {create_hash(s,q): i for i, s in enumerate(S)}
    g_map = {create_hash(g,q): i for i, g in enumerate(G)}
    prod_map = {} # Needed in the incidence checking
    for s in S:
        for g in G:
            prod_map[(create_hash(s, q), create_hash(g, q))] = create_hash(s @ g, q)
    print("Generating adjacency matrix...")
    adj = generate_adj_matrix(g_map, S, p, q)
    # If, because the calculation is immense.
    if RAMANUJAN_CHECK:
        print("Checking if (G,S) is Ramanujan...")
        if check_ramanujan_bound(adj,p):
            pass
        else:
            raise RuntimeError("(G,S) pair is not Ramanujan.")
    print(
f'''Ramanujan (G,S) created, |G|={len(G)}, |S|={len(S)}
with adjacency dim={adj.size}
''')
    # We have now succeeded in creating a Ramanujan graph
    # We shall now create the qLDPC of 2 distinct Tanner graphs
    # And combine these 2 with a lifted-product.
    E_horizontal, E_up, F, V = lifted_product_shuffle(g_map, s_map)
    r = p+1
    r_div =r//R_DIVISOR
    h = np.random.randint(0,2, size=(r_div, r))
    h_prime = np.random.randint(0, 2, size=(r_div, r))
    # This part could be optimized with numba or some other
    # Python JIT-compiler. This part takes a really long time...
    print(f'''
Generated basis elements:
E->  size: {len(E_horizontal)}
E_up size: {len(E_up)}
F    size: {len(F)}
V    size: {len(V)}''')
    print("Calculating stabilizer matrices H_X and dual_H_Z...")
    dual_H_Z_gen = gen_stabilizer_matrix(h, E_horizontal, E_up, F, V, prod_map)
    H_X_gen = gen_stabilizer_matrix(h_prime, E_horizontal, E_up, F, V, prod_map)

    dual_H_Z = create_stabilizer_matrix(dual_H_Z_gen)
    H_X = create_stabilizer_matrix(H_X_gen)

    print(f'''Generated stabilizers H_Z* and H_X. |H_Z*|={dual_H_Z.size}, |H_X|={H_X.size}.\n
Z: M_data {dual_H_Z_gen[2]}, \nN_data {dual_H_Z_gen[-1]}\n\n-------------------\n
X: M_data {H_X_gen[2]}, \nN_data {H_X_gen[-1]}''')
