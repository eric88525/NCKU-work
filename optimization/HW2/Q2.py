
from cmath import inf
from readline import set_startup_hook
from sympy import diff, symbols
import numpy as np
import math


# the equation should pass a and b symbols
def armijo(Xi: np.array, direction: np.array, equation) -> float:

    max_iter = 3
    # ε
    epsilon = 0.2
    a, b, lambda_ = symbols("a,b,lambda_")

    # x_i+1 = x_i + direction * lambda
    X_next = Xi + lambda_ * direction

    lambda_func = equation.subs([[a, X_next[0]], [b, X_next[1]]])
    diff_lambda_func = diff(lambda_func, lambda_)

    mse_func = lambda_func.subs(lambda_, 0) + \
        diff_lambda_func.subs(lambda_, 0)*epsilon*lambda_

    best_lambda = 1.0

    for iter in range(1, 100):
        func_val = lambda_func.subs(lambda_, best_lambda)
        F_val = mse_func.subs(lambda_, best_lambda)

        if func_val >= F_val:
            best_lambda /= 2.0
        else:
            if iter >= max_iter:
                return best_lambda
            best_lambda *= 2

    return best_lambda


# mse_func(x) = ax + b = y_pred
# we try to minimize  mean squred error(y_pred, y_label) = (mse_func(x) - y)^2 = (ax+b-y)^2
def fletcher_reeves(x: np.array, y: np.array, Xi: np.array, max_iter=1000):

    # ε
    epsilon = 0.01

    a, b = symbols("a, b")
    mse_func = ((a*x + b - y)**2).sum()

    gradient_func = [diff(mse_func, a), diff(mse_func, b)]

    for iter in range(1, max_iter+1):

        to_sub = [[a, Xi[0]], [b, Xi[1]]]
        # direction = -gradient(x_i)
        gradient = np.array([gradient_func[0].subs(
            to_sub), gradient_func[1].subs(to_sub)])
        direction = -1 * gradient

        if math.sqrt((direction**2).sum()) < epsilon:
            break

        # find best lambda by armijo
        lambda_ = armijo(Xi, direction, mse_func)

        # X_(i+1) = X_i + λ * direction
        Xi = Xi + lambda_ * direction

        to_sub = [[a, Xi[0]], [b, Xi[1]]]

        print(
            f"Iter {iter}: a = {Xi[0]:.3f}, b = {Xi[1]:.3f}, f(a, b) = {mse_func.subs(to_sub)/len(x):.3f}")

        gradient_next = np.array([gradient_func[0].subs(
            to_sub), gradient_func[1].subs(to_sub)])

        # cacualte next direction
        beta = (gradient_next.T @ gradient_next) / (gradient.T @ gradient)
        direction = -1 * gradient_next + beta * direction

        if math.sqrt((direction**2).sum()) < epsilon:
            break

    return Xi


def quasi_newton(x: np.array, y: np.array, Xi: np.array, degree: int = 1, epsilon: float = 0.001, max_iter: int = 100):

    a, b = symbols("a,b")

    #f(a, b) = (mx + b - y)^2.
    f = ((a * x + b - y) ** 2).sum()

    gradient_func = [diff(f, a), diff(f, b)]

    beta = np.identity(degree + 1)

    for step in range(max_iter):

        to_sub = [[a, Xi[0]], [b, Xi[1]]]
        df_val = np.array([gradient_func[0].subs(to_sub), gradient_func[1].subs(to_sub)])

        s = -1 * beta @ df_val

        if math.sqrt((s**2).sum()) < epsilon:
            break

        # Find proper lambda value.
        lambda_val = armijo(Xi, s, f)

        # Update Xi = (a , b).
        p_new = Xi + lambda_val * s
        to_sub = {a: p_new[0], b: p_new[1]}
        df_new_val = np.array([gradient_func[0].subs(to_sub), gradient_func[1].subs(to_sub)])

        d = p_new - Xi
        g = df_new_val - df_val
        d, g = d[:, None], g[:, None]

        beta = beta + (d @ d.T)/(d.T @ g) - \
            (beta @ g @ g.T @ beta)/(g.T @ beta @ g)
        s = -1 * beta @ df_new_val

        # Update Xi.
        Xi = p_new


        print(f"Iter: {step}\t Xi: {Xi}")

    return Xi



def main():

    x = np.array([0.1, 0.9, 1.9, 2.3, 3, 4.1, 5.2,
                  5.9, 6.8, 8.1, 8.7, 9.2, 10.1, 12])
    y = np.array([20, 24, 27, 29, 32, 37.3, 36.4,
                  32.4, 28.5, 30, 38, 43, 40, 32])

    start_point = np.array([1.0, 1.0])

    ans = fletcher_reeves(x, y, start_point, 1000)
    print(ans)

    print("\n\n")

    ans = quasi_newton(x, y, start_point)
    print(ans)


if __name__ == "__main__":
    # print(diff(func, a))
    main()
