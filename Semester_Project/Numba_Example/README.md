# Python Numba Example Directory

This directory is intended to aid you in downloading and using Python Numba in high performance computing! Below you will find installation instructions as well as example scripts to help you get an idea of what Numbas is all about!

## Table of Contents
- [Abstract](#abstract)
- [Installation](#installation)

## Abstract
Numba is a Just-In-Time (meaning your code is compiled during execution rather than before) Python compiler that works hand-in-hand with NumPy to optimize code that focuses on mathematical computations. It is primarily used in scientific and engineering applications to accelerate computational workloads, such as simulations, numerical solvers, and data analysis, by enabling high-performance execution of Python functions without requiring extensive rewrites or the more advanced debugging that come with using C/C++. Rather than a "middlewear" or standalone application, Numba functions as a programming tool meant to enhance your numerical code with minimal modifications. 

## Installation 
1. Using a development node of your choice on the HPCC, clone this repository using this command: `git clone https://github.com/wilcox63/CMSE401_SS25.git`.
2. Run the command `pip install numba` to get Numba downloaded!
3. Navigate to the `Numba_Example` folder using `cd` commands.
4. Using your text editor of choice, navigate to the sum_of_squares example file (`vim sum_of_powers.py`). Take a quick look at the code and comments to get an idea of what the example is doing.
5. Use the command `python sum_of_powers.py` to execute the file. You should be able to see the time it took the file to execute. Take mental note of this, now we can compare the time it takes to execute with Numba!
6. Similar to step 3, navigate to the `sop_Numba.py` file, Take a look at the code and the comments to gain a better understanding of how to integrate Numabe into your Python code.
7. Using the same command as step 5 (replacing the file name with `sop_Numba.py`), execute the code with Numba integration and compare the execution times. You should see a large speed-up!
8. To get the code running on the batch scheduler, first navigate to the `sop_Numa.sb` file as above, and take a look at the comments to understand what the SLURM submission script is doing.
9. Next, run the command `sbatch sop_Numba.sb` to submit the job script.

## References
- [Sum of Squares Example](https://www.geeksforgeeks.org/python-program-for-sum-of-squares-of-first-n-natural-numbers/#) - Please note this example was used as a reference and I updated it to better fit the goal of this example.
- [Numba Documentation](https://numba.pydata.org/)
