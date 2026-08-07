import random
from sympy.solvers.diophantine.diophantine import sum_of_squares
from sympy import isprime
from itertools import permutations, product
import numpy as np
import math
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
            return p, q

# we need an integer u that satisfies u**2 = -1 mod q
def find_u(q):
    while True:
        a = random.randint(2,q-1)   # pick some random integer
        u = pow(a, int((q-1)/4), q) # check if it satisfies the above condition, if not, pick another a
        if u**2 % q == q-1:
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
    return solutions

# find the elements of S
def generate_S(solutions, u, p, q):  
    S = []
    for sol in solutions:
        v = np.zeros((2,2))
        v[0,0] += p**(-1/2)*((sol[0] + u*sol[1]) % q)
        v[0,1] += p**(-1/2)*((sol[2] + u*sol[3])) % q
        v[1,0] += p**(-1/2)*((-sol[2] + u*sol[3])) % q
        v[1,1] += p**(-1/2)*((sol[0] - u*sol[1])) % q
        S.append(v)
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
    return list(set(G))  # dedupe the list so we are left with mod +-identity. this final list should have q(q^2-1)/2 elements.

# generate the edges of our graph
def generate_edges(G,S,q):
    E = []
    explicit_E = []
    for i in range(len(G)):  
        G[i] = np.reshape(np.array(G[i]), (2,2))  # turn our list of tuples into 2x2 matrices to perform calculations
    for s_matrix in S:
        for g_matrix in G:
            E.append((g_matrix, s_matrix))  # we can safely do both in the same loop as this first append takes a fraction of the time it takes for the second
            explicit_E.append(((0,g_matrix),(1, (s_matrix @ g_matrix) % q)))  # operating with s onto g keeps the result in G
    return E, explicit_E

def test(G, S, q):
    for i in range(len(G)):  
        G[i] = np.reshape(np.array(G[i]), (2,2))  # turn our list of tuples into 2x2 matrices to perform calculations
    for s_matrix in S:
        for g_matrix in G:
            to_find = (s_matrix @ g_matrix) % q
            for i in range(len(G)):
                if np.array_equiv(to_find, G[i]):
                    print(i)

def main():
    primes = prime_finder()
    p = primes[0]
    q = primes[1]
    print(f'\np = {p}\nq = {q}')

    u = find_u(q)
    print(f'u = {u}')

    decomposition = decompose_p(p)

    S = generate_S(decomposition, u, q)
    G = generate_G(q)
    print(f'|S| = {len(S)} (should be p+1 = {p+1})\n|G| = {len(G)} (should be q*(q**2-1)/2 = {int(q*(q**2-1)/2)})')

#    edges = generate_edges(G,S,q)
main()
