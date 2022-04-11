from typing import Callable
from .one_dim import golden_search
import random


def hookjeeve_method(x1_range: list, x2_range: list, target_func: Callable[[float, float], float]):
    # random init x0
    x1 = random.uniform(x1_range[0], x1_range[1])
    x2 = random.uniform(x2_range[0], x2_range[1])
    epsilon = 1e-7
    x_list, y_list = [], []
    step_size = [0.001, 0.001]
    prev_fx = float('inf')

    for i in range(100):
        # test x1
        fx1 = target_func(x1, x2)
        x1_test_add = target_func(x1 + step_size[0], x2)
        x1_test_minus = target_func(x1 - step_size[-1], x2)
        x1_next = x1

        if x1_test_add < x1_test_minus and x1_test_add < fx1:
            x1_next = x1 + step_size[0]
        elif x1_test_minus < x1_test_add and x1_test_minus < fx1:
            x1_next = x1 - step_size[0]

        x_list.append([x1_next, x2])
        y_list.append(target_func(x1_next, x2))

        # test x2
        fx2 = target_func(x1_next, x2)
        x2_test_add = target_func(x1_next, x2 + step_size[1])
        x2_test_minus = target_func(x1_next, x2 - step_size[1])
        x2_next = x2

        if x2_test_add < x2_test_minus and x2_test_add < fx2:
            x2_next = x2 + step_size[0]
        elif x2_test_minus < x2_test_add and x2_test_minus < fx2:
            x2_next = x2 - step_size[0]

        x_list.append([x1_next, x2_next])
        y_list.append(target_func(x1_next, x2_next))

        s1, s2 = x1_next-x1, x2_next-x2

        if s1 == s2 and s1 == 0:
            break
        # check range first
        while x1_range[0] <= x1_next + s1 <= x1_range[1] and \
                x2_range[0] <= x2_next + s2 <= x2_range[1]:
            # walk along direction until stop improve
            if target_func(x1_next + s1, x2_next + s2) < target_func(x1_next, x2_next):
                x1_next += s1
                x2_next += s2
            else:
                break

        x1, x2 = x1_next, x2_next
        x_list.append([x1, x2])
        y_list.append(target_func(x1, x2))

        # break condition |( f(x*) – q(x*) )/f(x*) | < 10-7
        if abs((prev_fx - y_list[-1])/prev_fx) <= epsilon:
            break
        prev_fx = y_list[-1]

    return x_list, y_list

def powell_method(x1_range: list, x2_range: list, target_func: Callable[[float, float], float]):

    # random x0
    x1 = random.uniform(x1_range[0], x1_range[1])
    x2 = random.uniform(x2_range[0], x2_range[1])
    epsilon = 1e-7
    x_list = []
    y_list = []
    prev_fx = float('inf')

    for i in range(100):
        # find best lambda and update x_1 = x_1 + lambda * s1
        x1_next = golden_search(
            x1_range[0], x1_range[1], lambda a: target_func(a, x2))[0][-1]
        x_list.append([x1_next, x2])
        y_list.append(target_func(x1_next, x2))

        # find best lambda and update x_2 = x_2 + lambda * s2
        x2_next = golden_search(
            x2_range[0], x2_range[1], lambda a: target_func(x1_next, a))[0][-1]
        x_list.append([x1_next, x2_next])
        y_list.append(target_func(x1_next, x2_next))

        # get direction
        s1 = x1_next - x1
        s2 = x2_next - x2

        # find lambda to minimize f(X + lambda*S)
        lambda_max, lambda_min = float('inf'), float('-inf')
        lambda_max = min((x1_range[1]-x1_next)/(s1+epsilon), (x2_range[1]-x2_next)/(s2+epsilon))
        lambda_min = max((x1_range[0]-x1_next)/(s1+epsilon), (x2_range[0]-x2_next)/(s2+epsilon))

        if lambda_max <= lambda_min:
            _lambda = 0
        else:
            _lambda = golden_search(lambda_min, lambda_max, lambda a: target_func(
                x1_next + a*s1, x2_next + a*s2))[0][-1]

        x1 = x1_next + _lambda*s1
        x2 = x2_next + _lambda*s2
        x_list.append([x1, x2])
        y_list.append(target_func(x1, x2))

        # break condition |( f(x*) – q(x*) )/f(x*) | < 10-7
        if abs((prev_fx - y_list[-1])/prev_fx) <= epsilon:
            break
        prev_fx = y_list[-1]

    return x_list, y_list
