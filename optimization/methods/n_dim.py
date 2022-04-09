from typing import Callable
from .one_dim import golden_search
import random


def hookjeeve_method(x1_range: list, x2_range: list, target_func: Callable[[float, float], float]):
    epsilon = 1e-5

    x1 = random.uniform(x1_range[0], x1_range[1])
    x2 = random.uniform(x2_range[0], x2_range[1])

    x1_list = [x1]
    x2_list = [x2]

    for i in range(100):

        x1_next = golden_search(
            x1_range[0], x1_range[1], lambda a: target_func(a, x2))[0][-1]
        x2_next = golden_search(
            x2_range[0], x2_range[1], lambda a: target_func(x1_next, a))[0][-1]
        # d = x_k+1 - x_k
        dx1 = x1_next - x1
        dx2 = x2_next - x2

        if (dx1**2 + dx1**2)**0.5 <= epsilon:
            break

        while x1_range[0] <= x1_next and x1_next <= x1_range[1] and x2_range[0] <= x2_next and x2_next <= x2_range[1]:
            if target_func(x1_next + dx1, x2_next + dx2) < target_func(x1_next, x2_next):
                x1_next += dx1
                x2_next += dx2
            else:
                break
        x1 = x1_next
        x2 = x2_next
        x1_list.append(x1)
        x2_list.append(x2)

    return x1_list, x2_list
