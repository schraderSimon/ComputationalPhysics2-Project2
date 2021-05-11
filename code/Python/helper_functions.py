import numpy as np
from numba import jit, njit
@jit(nopython=True)
def integrate_trapezoidal(function_array,step):
    """Simple trapezoidal rule"""
    integral=0
    for i in range(1,len(function_array)-1):
        integral+=function_array[i]
    integral*=2
    integral+=function_array[0]+function_array[-1]
    integral*=step/2
    return integral
