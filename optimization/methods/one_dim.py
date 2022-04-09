from typing import Callable

def golden_search(range_min: float, range_max: float, target_func: Callable[[float], float]):

    if range_max < range_min:
        return [[],[]]

    # golden ration
    golden_ratio = 1.618
    epsilon = 1e-5
    x_list, y_list = [], []

    # next range
    d = (range_max - range_min) / golden_ratio
    x1 = range_max - d
    x2 = range_min + d

    # range_min---x1---x2---range_max
    for i in range(100):

        if range_max-range_min < epsilon:
            break

        # compare function value and update range
        f_x1 = target_func(x1)
        f_x2 = target_func(x2)

        if f_x2 > f_x1:
            range_max = x2
            x2 = x1
            x1 = range_max - (range_max - range_min) / golden_ratio
        else:
            range_min = x1
            x1 = x2
            x2 = range_min + (range_max - range_min) / golden_ratio

        x_list.append((range_max+range_min)/2)
        y_list.append(target_func((range_max+range_min)/2))

    return [x_list, y_list]

def fibonacci_search(range_min: float, range_max: float, target_func: Callable[[float], float]):

    epsilon = 1e-5
    fibonacci_list = [1, 1]
    x_list, y_list = [], []

    while fibonacci_list[-1] < (range_max-range_min)/epsilon:
        fibonacci_list.append(fibonacci_list[-2] + fibonacci_list[-1])

    x1 = range_min + (fibonacci_list[-3] /
                      fibonacci_list[-1]) * (range_max-range_min)
    x2 = range_min + (fibonacci_list[-2] /
                      fibonacci_list[-1]) * (range_max-range_min)

    # range_min---x1---x2---range_max
    for n in range(len(fibonacci_list)-2, 1, -1):

        f_x1 = target_func(x1)
        f_x2 = target_func(x2)

        if f_x2 > f_x1:
            range_max = x2
            x2 = x1
            x1 = range_min + \
                (fibonacci_list[n-2] / fibonacci_list[n]) * \
                (range_max-range_min)
        else:
            range_min = x1
            x1 = x2
            x2 = range_min + \
                (fibonacci_list[n-1] / fibonacci_list[n]) * \
                (range_max-range_min)

        x_list.append((range_max+range_min)/2)
        y_list.append(target_func((range_max+range_min)/2))

    return [x_list, y_list]