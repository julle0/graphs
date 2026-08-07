import random
from sympy.solvers.diophantine.diophantine import sum_of_squares
from sympy import isprime
from itertools import permutations, product
import numpy as np
import math
import time

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
    n = 1
    while np.sqrt(p % q + q*n) != int(np.sqrt(p % q + q*n)):
        n += 1
    print(n, p % q + q*n)

    for sol in solutions:
        v = np.zeros((2,2))
        v[0,0] += ((sol[0] + u*sol[1])) % q
        v[0,1] += ((sol[2] + u*sol[3])) % q
        v[1,0] += ((-sol[2] + u*sol[3])) % q
        v[1,1] += ((sol[0] - u*sol[1])) % q
        S.append(v)
    return S

def find_u(q):
    while True:
        a = random.randint(2,q-1)   # pick some random integer
        u = pow(a, int((q-1)/4), q) # check if it satisfies the above condition, if not, pick another a
        if u**2 % q == q-1:
            return u

p = 13
q = 17
u = find_u(q)
sols = decompose_p(p)
S = generate_S(sols, u ,p, q)
