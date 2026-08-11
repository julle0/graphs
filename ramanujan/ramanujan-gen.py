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
def prime_finder():
    p = int(input('Pick initial value to start search: '))  # the graph will be (p+1)-regular and have q*(q**2-1) vertices
    q_list = []
    while not (isprime(p) and p%4 == 1):  # simpler case: p only needs to be prime and congruent to 1 mod 4 
        p += 1  # difference between values of successive primes (approaches ln(p) for large p) is small enough for our purposes to perform a dumb search such as this one
    for iteration in range(5):  # find the 5 smallest candidates for q
        if iteration == 0:  # start the search from the smallest integer greater than 2*p**(1/2)
            q_candidate = math.ceil(2*p**(1/2))  
        else:  # continue the search from the end of the list
            q_candidate = q_list[-1]+1
        while not (
                isprime(q_candidate)  # q needs to also be prime but also: 
                and q_candidate%4 == 1  # congruent to 1 mod 4
                and pow(p, (q_candidate-1)//2, q_candidate) == 1  # p**((q-1)/2) = 1 mod q (in order for the generating set to land in PSL(2, F_q))
                ): 
            q_candidate += 1
        q_list.append(q_candidate)
    while True:
        q = int(input(f'p = {p}, pick a value for q from {q_list}: '))
        if q not in q_list:
            print('Choice not in list, pick again.')
            continue
        else:
            print(f'Primes found succesfully, p = {p}, q = {q}')
            return p, q

# we need an integer u that satisfies u**2 = -1 mod q
def find_u(q):
    while True:
        a = random.randint(2,q-1)   # pick some random integer
        u = pow(a, int((q-1)/4), q) # check if it satisfies the above condition, if not, pick another a
        if u**2 % q == q-1:
            print(f'u found succesfully, u = {u}')
            return u

# we need to find a sort of decomposition of p such that a**2 + b**2 + c**2 + d**2 = p with a > 0 and b,c,d even
def decompose_p(p):
    solutions = []
    for sol in sum_of_squares(p,4, zeros=True):  # use sympys sum_of_squares() function to find such solutions. this function only gives the non-permutated/cleaned-up solutions
        count = sum(1 for i in sol if i % 2 == 0)
        if count == 3:  # count == 3 is the only relevant solution since p is a prime, so count == 4 is impossible
            sol_list = sorted(sol, key = lambda x: 1 - x%2) # sort by odd and even numbers, odd first for a
            for perm in set(permutations(sol_list[1:])):  # permutate the last 3 indices while keeping the first one the same since it is the odd a
                # this sign implementation (52-57) was essentially directly copied from claude, i had no idea how to do it
                nonzero_ind = [i for i, v in enumerate(perm) if v != 0]  # find the nonzero indices
                for sign in product([1,-1], repeat = len(nonzero_ind)):  # solution may include negative numbers, so iterate over multiplying by 1 or -1
                    perm_list = list(perm)  # permutations() returns a tuple, but it is convenient to turn this into a list
                    for i, s in zip(nonzero_ind, sign):
                        perm_list[i] *= s
                    perm_list.insert(0,sol_list[0])  # insert a into the permutated list
                    solutions.append(perm_list)  # append to solutions
    print(f'Solutions found succesfully')
    return solutions

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
            raise ValueError(f'Something went wrong, det({v})%{q} = {np.linalg.det(v)%q}, not 1')
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
    errors = []
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

def ramanujan_bound(adj_matrix, p):
    t0 = time.time()
    eigvals = list(eigsh(adj_matrix, k=6, which='LM', return_eigenvectors=False))
    eigvals = [np.round(abs(x), decimals=5) for x in eigvals]
    eigvals = [x for x in eigvals if x != p + 1]
    bound = 2*np.sqrt(p)
    l = max(eigvals)
    verdict = 'is' if l <= bound else 'is not'
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
