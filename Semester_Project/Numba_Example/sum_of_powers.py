import numpy as np

import numpy as np

import time

'''

This script is a Python computational example that uses NumPy arrays to generate a large amount of

random data, and compute a sum of increasing powers.  

'''



def sum_of_squares(arr, power=2):

    total = 0.0

    # Iterate over the array, make the computation for each element, and add it to the total.

    for x in range(len(arr)):

        total += arr[x] ** min(power, 10) 

        power += 1

    return total



size = 5 * 10**7  # 50 million elements

data = np.random.randint(1, 1000, size, dtype=np.int64).astype(np.float64) 



# Time the computation for comparison with use of Numba

start_time = time.time()

result = sum_of_squares(data)

end_time = time.time()



print(f"Sum of powers: {result:.2f}")

print(f"Execution time: {end_time - start_time:.4f} seconds")
