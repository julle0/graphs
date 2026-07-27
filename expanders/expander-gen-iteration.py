import numpy as np
import random
import sys

# For aesthetic purposes
def delete_last_line():
    # Cursor up one line
    sys.stdout.write('\x1b[1A')

    # Delete last line
    sys.stdout.write('\x1b[2K')

def adj_matrix_gen(c, d, n):
#   Constructing an adjacency matrix for an expander graph:
#       Parameters c, d, n:
#
#       An expander graph must be (c,d) regular, i.e.
#       a larger vertex set in your graph will be c-regular (all vertices have c edges)
#       and a smaller vertex set will be d-regular
#       
#       c and d must fulfill the following:
#           c < d
#           2d-n > 0
#           (c/d)*n is a positive integer
#
#       We wish to represent this graph as a matrix:
#           Each row will represent a message bit (n rows)
#           Each column will represent a parity check vertex/bit ((c/d)*n columns)
#           Each entry [i,j] tells you if a message bit i is connected to a parity check bit j
#       e.g. if the first row of the matrix reads [[1,1,1,0],...], then the message bit n_0 is connected to parity check bits p_0,p_1,p_2
#
#       The values of c and d impose restrictions on the rows and columns:
#           Each row may only have weight c and each column may only have weight d

    # Errors to enforce the restrictions
    if c >= d:
        raise ValueError(f'{c}>={d}, c must be strictly less than d, otherwise not an expander')
    
    if 2*d-n < 0:
        raise ValueError(f'2*d-n = {2*d-n} must be greater than zero, otherwise impossible to fill matrix')
    
    if int((c/d)*n) != (c/d)*n:
        raise ValueError(f'(c/d)*n = {(c/d)*n} must be an integer, matrix indices cant be floats')

    # Defining variables as per explanation
    cols = int((c/d)*n)
    rows = n
    adj_matrix = np.zeros([rows, cols], dtype=int)

    # Preparing column weight calculations
    col_weight = np.zeros(cols, dtype=int)

    succ = 0
    tries = 100000
    # Iteration loop
    for _ in range(tries):
        adj_matrix = np.zeros([rows, cols], dtype=int)
        col_weight = np.zeros(cols, dtype=int)
        col_check = 0
        row_weight = np.zeros(rows, dtype=int)
        row_check = 0
        # Main loop
        for i in range(rows):
            # Find which columns have a weight less than the limit
            available = []
            for j in range(cols):
                if col_weight[j] < d:
                    available.append(j)

            # Find whether there are enough columns to fulfill the row weight requirement
            if len(available) < c:
                continue

            # Creating the first row
            if i == 0:
                random.shuffle(available)
                chosen = available[:c]
                for j in chosen:
                    adj_matrix[i, j] += 1
                    col_weight[j] += 1
            # The rest of the rows
            else:
                available.sort(key=lambda j: (col_weight[j], random.random())) # Sorts the available columns by weight and breaks ties with random.random()
                chosen = available[:c] # Pick out the first c elements from this list
                for j in chosen: # Add ones to the desired spots
                    adj_matrix[i, j] += 1
                    col_weight[j] += 1

            row_weight[i] += sum(adj_matrix[i]) # Calculate row weight

        row_check = sum(1 for j in row_weight if j == c) # Check whether the matrix fulfills our requirements
        col_check = sum(1 for j in col_weight if j == d)

        # Count successful matrices
        if row_check == rows and col_check == cols:
            delete_last_line()
            print(f'{(_/tries)*100:.1f}% Successes: {succ}')
            succ += 1
        else:
            delete_last_line()
            print(f'{(_/tries)*100:.1f}% Successes: {succ}')
            continue

        delete_last_line()

    return print(f'Successes: {succ}\n{(succ/tries)*100}% of total\n{adj_matrix}')
        


adj_matrix_gen(12,24,32)
