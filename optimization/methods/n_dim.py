from typing import Callable
from .one_dim import golden_search
import sys
import random


def hookjeeve_method(x1_range: list, x2_range: list, target_func: Callable[[float, float], float]):
    epsilon = 1e-5
    x1 = random.uniform(x1_range[0], x1_range[1])
    x2 = random.uniform(x2_range[0], x2_range[1])

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

        lambda_max, lambda_min = sys.float_info.max, sys.float_info.min

        if dx1 >= 0:
            lambda_max = min(lambda_max, min(
                x1_range[1]-x1_next, x1_next-x1_range[0]) / abs(dx1))
        else:
            lambda_min = max(lambda_min, max(
                x1_next-x1_range[1], x1_range[0]-x1_next)/abs(dx1))

        if dx2 >= 0:
            lambda_max = min(lambda_max, min(
                x2_range[1]-x2_next, x2_next-x2_range[0]) / abs(dx2))
        else:
            lambda_min = max(lambda_min, max(
                x2_next-x2_range[1], x2_range[0]-x2_next)/abs(dx2))

        print(
            f"x1 {x1:.7f} x1_next {x1_next:.7f} dx1 {dx1:.7f} \n x2 {x2:.7f} x2_next {x2_next:.7f} dx2 {dx2:.7f} \n lmax {lambda_max:.7f} lmin {lambda_min:.7f} ")

        _lambda, _ = golden_search(
            lambda_min, lambda_max, lambda a: target_func(x1_next+a*dx1, x2_next+a*dx2))

        if len(_lambda) > 0:
            _lambda = _lambda[-1]
        else:
            _lambda = 0

        x1 = x1_next + _lambda * dx1
        x2 = x2_next + _lambda * dx2

        print(f"x1 {x1:.5f} x2 {x2:.5f} f(x) = {target_func(x1, x2):.5f}")
        print("---------------------------------------------\n\n")