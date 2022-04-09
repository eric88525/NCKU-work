from typing import Callable
from methods.n_dim import hookjeeve_method

def himmelblau_func(x1: float, x2: float) -> float:
    return (x1*x1+x2-11)**2 + (x1+x2*x2 -7)**2

def main():
    hookjeeve_method([-5,5],[-5,5],himmelblau_func)
    #print( himmelblau_func(-3.779310252732609, -3.283186703148891) )


if __name__ == "__main__":
    main()
