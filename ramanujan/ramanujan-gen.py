import random
from sympy.solvers.diophantine.diophantine import sum_of_squares
from sympy import isprime
from itertools import permutations

# Based on the LPS construction of Ramanujan graphs
# find two primes p and q with conditions specified below:
def prime_finder(p): 
    while not (isprime(p) and p%4 == 1):  # simpler case: p only needs to be prime and congruent to 1 mod 4 
        p += 1
    q = p + 1
    while not (
            isprime(q)  # q needs to also be prime but also: 
            and q%4 == 1  # congruent to 1 mod 4
            and q > 2*p**(1/2)  # greater than 2*sqrt(p)
            and pow(p, (q-1)//2, q) == 1  # p**((q-1)/2) = 1 mod q (in order for the generating set to land in PSL(F_q))
               ):
        q += 1
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
        count = 0
        for i in sol:  # check which solutions satisfy the above conditions. a count of >= 3 means there are at least 3 even numbers
            if i % 2 == 0:
                count += 1
        if count >= 3:
            sol_list = sorted(sol, key = lambda x: 1 - x%2) # sort by odd and even numbers, odd first for a
            solutions.append(sol_list)  # append the non-permutated solutions
            for perm in permutations(sol_list[1:]):  # permutate the last 3 indices while keeping the first one the same since it is the odd a
                perm = list(perm)  # permutations() returns a tuple, but it is convenient to turn this into a list
                perm.insert(0,sol_list[0])  # insert a into the permutated list
                solutions.append(perm)  # append to solutions
    return solutions

# TO ADD:
# CASE WHERE COUNT = 4
# SIGN FLIPS
