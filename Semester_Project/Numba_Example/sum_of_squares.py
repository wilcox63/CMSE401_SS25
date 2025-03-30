import numpy as np

import time

'''
This script is a Python computational example that uses NumPy arrays to generate a large amount of
random data, and compute its sum of squares. 
'''

def sum_of_squares(arr):

    total = 0.0
    #Iterate over the array, compute the square of each element, and add it to the total.
    for x in range(len(arr)):

        total += arr[x] ** 2

    return total



# Generate a large array of random numbers

size = 10**7  # 10 million elements

data = np.random.randint(1,20,size)


#Time the computation for comparison with use of Numba
start_time = time.time()

result = sum_of_squares(data)

end_time = time.time()



print(f"Sum of squares: {result:.2f}")

print(f"Execution time: {end_time - start_time:.4f} seconds")
