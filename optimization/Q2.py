from typing import Callable
from methods.n_dim import hookjeeve_method
import matplotlib.pyplot as plt


def himmelblau_func(x1: float, x2: float) -> float:
    return (x1*x1+x2-11)**2 + (x1+x2*x2 - 7)**2


def main():
    x1_list, x2_list = hookjeeve_method([-5, 5], [-5, 5], himmelblau_func)
    fig, ax = plt.subplots()
    ax.plot(x1_list, x2_list, '-o', ms=5, lw=2, alpha=0.7, mfc='orange')
    ax.grid()

    plt.axis([-5, 5, -5, 5])
    plt.text(x1_list[0], x2_list[0], "start")
    plt.text(x1_list[-1], x2_list[-1], "end")
    plt.title(
        f"hookjeeve: f({x1_list[-1]:.6f}, {x2_list[-1]:.6f}) = {himmelblau_func(x1_list[-1], x2_list[-1]):.3f}")
    plt.savefig('hookjeeve_method.png')


if __name__ == "__main__":
    main()
