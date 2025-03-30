import os

os.environ["NUMBA_THREADING_LAYER"] = "omp" 
import numpy as np
import time
from numba import njit, prange

'''
This script is a Python computational example that uses NumPy arrays to generate a large amount of
random data, and compute a sum of increasing powers - utilizing Python Numba. 
'''

#njit loads the usage of Numba, parallel = true enables the use of parallel processing to speed up 
#computation.
@njit(parallel=True)
def sum_of_squares(arr, power=2):
    total = 0.0
    # Iterate over the array, make the computation for each element, and add it to the total.
    for x in prange(len(arr)):  # Use prange for parallel execution
        total += arr[x] ** min(power + x, 10) 
    return total

# Generate a large array of random numbers as float64 to prevent overflow
size = 5 * 10**7  # 50 million elements
data = np.random.randint(1, 1000, size, dtype=np.int64).astype(np.float64) 

# Time the computation for comparison
start_time = time.time()
result = sum_of_squares(data)
end_time = time.time()

print(f"Sum of powers: {result:.2f}")
print(f"Execution time: {end_time - start_time:.4f} seconds (with Numba)")
