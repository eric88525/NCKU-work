import math
from methods.one_dim import golden_search, fibonacci_search
import matplotlib.pyplot as plt


def func(x: float) -> float:
    return (x**3) * (math.exp(-1*x**2))


def draw_plt(x_list: list, y_list: list, method_name: str):
    plt.style.use('fivethirtyeight')
    plt.xlabel("Iteration number")
    plt.ylabel("Function value")
    plt.plot([i for i in range(len(x_list))], y_list, '.')
    plt.title(f"{method_name} min: f({x_list[-1]:.6f}) = {y_list[-1]:.3f}")
    plt.savefig(f"{method_name}.png")
    plt.show()


def main():

    # GOLDEN
    x_list, y_list = golden_search(-2, 2, func)
    draw_plt(x_list, y_list, method_name="Golden")

    # FIBONACCI
    x_list, y_list = fibonacci_search(-2, 2, func)
    draw_plt(x_list, y_list, method_name="Fibonacci")


if __name__ == "__main__":
    main()
