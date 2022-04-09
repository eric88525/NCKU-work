import math
import matplotlib.pyplot as plt
from methods.one_dim import golden_search, fibonacci_search
from methods.draw import draw_plt


def func(x: float) -> float:
    return (x**3) * (math.exp(-1*x**2))


def main():

    # GOLDEN
    x_list, y_list = golden_search(-2, 2, func)
    draw_plt(x_list, y_list, [-2, 2], [-0.5, 0.5], method_name="GOLDEN")

    # FIBONACCI
    x_list, y_list = fibonacci_search(-2, 2, func)
    draw_plt(x_list, y_list, [-2, 2], [-0.5, 0.5], method_name="FIBONACCI")


if __name__ == "__main__":
    main()
