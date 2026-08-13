import random
from sympy.solvers.diophantine.diophantine import sum_of_squares
from sympy import isprime
from itertools import permutations, product
import numpy as np
import math
from scipy.sparse.linalg import eigsh
import time

# Based on the LPS construction of Ramanujan graphs
# find two primes p and q with conditions specified below:
def prime_finder(p_range):
    init_rand = random.randint(p_range[0], p_range[1])
    step = 2
    p = init_rand+1 if init_rand % 2 == 0 else init_rand # Step is a even num
    while not (isprime(p) and p%4 == 1):
        p += step
    # Next, we will find the prime q:
    q_init = math.ceil(2*p**(1/2))
    q =  q_init+1 if q % 2 == 0 else q_init
    while not (isprime(q) and q%4 == 1 and q**((q-1)//2) % q  == 1):
        q += step
    return p, q

# we need an integer u that satisfies u**2 = -1 mod q
def find_u(q):
    for i in range(2,q-1):
        u = i**(int(q-1)/2) % q
        if u == q-1:
            return u
    raise RuntimeError("No such u exists.")

def decompose_p(p):
    sol = []
    for _sol in sum_of_squares(p,4,zeros=True):
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
def generate_S(solutions, u, p, q):
    S = []  # before calculating the matrices, we need to find a normalisation constant to ensure that all determinants are 1 mod q
    n = 1
    while np.sqrt(p % q + q*n) != int(np.sqrt(p % q + q*n)):  # look for an integer which is congruent to p mod q but also an integer when we take its square root
        n += 1
    norm_const = (int(np.sqrt(p % q + q*n))) % q
    inv_norm_const = 1
    while (norm_const*inv_norm_const) % q != 1:  # find the multiplicative inverse of this integer mod q
        inv_norm_const += 1
    for sol in solutions:
        v = np.zeros((2,2))  # calculate the matrix entries which are of form
        v[0,0] += (inv_norm_const*(sol[0] + u*sol[1])) % q  # a+ub 
        v[0,1] += (inv_norm_const*(sol[2] + u*sol[3])) % q  # c+ud 
        v[1,0] += (inv_norm_const*(-sol[2] + u*sol[3])) % q  # etc. where a,b,c,d come from the decomposition of p and u from find_u()
        v[1,1] += (inv_norm_const*(sol[0] - u*sol[1])) % q
        S.append(v)
    for v in S:  # check that all matrices have det 1 mod q to a certain tolerance since numpy creates rounding errors
        det = np.linalg.det(v) % q
        if abs(det-1) > 0.001:
            raise ValueError(f'Something went wrong, det({v})%{q} = {det}, not 1')
    if len(S) != p+1:
        raise ValueError(f'Length of S != {p + 1} but {len(S)}')
    print(f'S generated succesfully, |S| = {len(S)}')
    return S

# generate G = PSL(2, F_q)
def generate_G(q):
    G = []  # elements of G need to have determinant = 1 and an entry's inverse mod q may not be present,
    seen = []  # or in other words, all of PSL(2, F_q) modulo +-the identity matrix
    for i in range(q**4):  # PSL(2,F_q) contains all 2x2 matrices with entries in F_q, so q**4 unique matrices
        digits = []  # turn base 10 integers in q**4 into base q representation
        while i:  # this loop is based on one i found on stackexchange 
            digits.append(int(i % q))
            i //= q
        if len(digits) < 4:  # pad out shorter digits with zeros since we always need 4 matrix entries (currently handled as lists/tuples)
            for _ in range(4-len(digits)):
                digits.append(0)
        if (digits[0]*digits[3]-digits[1]*digits[2]) % q == 1:  # enforce the det = 1 condition
            seen.append(tuple(digits))  # seen represents all of PSL(2, F_q) with det = 1
    for dts in seen:  # reducing the list so, that each matrix and its inverse pair gets turned into a pair of either itself or its inverse
         G.append(min(dts, tuple((-x)%q for x in dts)))  # deterministically pick either the original or the inverse, now governed by taking min() of two tuples
    G = list(set(G))
    if len(G) != q*(q**2-1)/2:
        raise ValueError(f'Length of G != {q*(q**2-1)/2} but {len(G)}')
    print(f'G generated succesfully, |G| = {len(G)}')
    return G  # dedupe the list so we are left with mod +-identity. this final list should have q(q^2-1)/2 elements.

# generate the edges of our graph
def generate_edges(G,S,q):
    E = []
    explicit_E = []
    for i in range(len(G)):  
        G[i] = np.reshape(np.array(G[i]), (2,2))  # turn our list of tuples into 2x2 matrices to perform calculations
    for s in S:
        for g in G:  # we can safely do both appendings in the same loop as this first append takes a fraction of the time it takes for the second
            E.append((g, s))  # implicit format: contains all of the same information as the latter appending, but lighter
            explicit_E.append(((0,g),(1, (s @ g) % q)))  # operating with s onto g keeps the result in G
    return E, explicit_E

# generate the adjacency matrix
def generate_adj_matrix(G, S, p, q):  # we have G and S: each g is connected to an sg which stays in G. g is a matrix, but used as a label for the vertices. 
    n = len(G)
    adj_matrix = np.zeros((n,n), dtype = np.int8)  # n x n matrix where each entry describes a connection between two vertices 
    index = {g: i for i, g in enumerate(G)}  # enumerate G for a hash lookup
    G_matrices = [np.reshape(np.array(g), (2,2)) for g in G]  # turn the tuples into matrices for calculations
    entry_track = []
    for s in S:  
        for i, g in enumerate(G_matrices):  # enumerate the matrices for indexing the adjacency matrix
            prod = (s @ g) % q  # find the label of the matrix which is connected to g
            flat_prod = tuple(prod.flatten())  # turn the matrix into a tuple to perform upcoming min() tiebreaker since the product could land at an inverse we excluded in the construction of G
            canonical_prod = min(flat_prod, tuple((-x)%q for x in flat_prod))  # the tiebreaker
            j = index[canonical_prod]  # use the lookup we created earlier to find the index of the result of the product
            if adj_matrix[i,j] > 0:
                entry_track.append((i,j))
            adj_matrix[i,j] += 1  # add a one to [i,j] to represent a connection with the ith entry and jth entries of G
    col_sums = [col_sum for col_sum in adj_matrix.sum(0)]
    row_sums = [row_sum for row_sum in adj_matrix.sum(1)]
    errors = []  # check for errors during the generation
    if entry_track:
        errors.append(f'Vertices connected more than once at {entry_track}')
    if not all(c == p + 1 for c in col_sums):
        errors.append(f'Column weight not consistently p + 1 = {p+1}: {col_sums}')
    if not all(r == p + 1 for r in col_sums):
        errors.append(f'Row weight not consistently p + 1 = {p+1}: {row_sums}')
    if errors:
        raise ValueError('\n'.join(errors))
    print(f'Adjacency matrix generated succesfully')
    return adj_matrix

# find out if our graph is truly Ramanujan
def ramanujan_bound(adj_matrix, p):  # a graph is Ramanujan if its adjacency matrix's largest absolute eigenvalue distinct from abs(p+1) is smaller than 2*sqrt(p)
    t0 = time.time()
    eigvals = list(eigsh(adj_matrix, k=4, which='LM', return_eigenvectors=False))  # find the k largest eigenvalues (keep in mind odd values for k are slower for some reason)
    eigvals = [np.round(abs(x), decimals=5) for x in eigvals]  # take absolute value and round to 5 decimals since the previous calculation introduces floating point errors
    eigvals = [x for x in eigvals if x != p + 1]  # remove all entries equal to p+1
    bound = 2*np.sqrt(p)
    l = max(eigvals)
    verdict = 'is' if l <= bound else 'is not'  # determine if our graph is Ramanujan
    print(f'Graph {verdict} Ramanujan, lambda(X) = {l} <= {bound} = 2*sqrt(p)')
    t1 = time.time()
    print(t1-t0)
    return l


def main():
    primes = prime_finder()
    p = primes[0]
    q = primes[1]
    u = find_u(q)

    decomposition = decompose_p(p)

    S = generate_S(decomposition, u, p, q)
    G = generate_G(q)
    
    adj = generate_adj_matrix(G, S, p, q)
    ramanujan_bound(adj,p)
main()
