import numpy as np
import matplotlib.pylab as plt
import math


# mean squared error of y_pred and y_label
def MSE(y_pred: np.array, y_label: np.array)->float:
    return ((y_pred-y_label)**2).sum()/len(y_label)


def find_coef(x:np.array, y:np.array)->np.array:

    matrix = [np.ones(len(x))]
    matrix.append([math.sin(_) for _ in x])
    matrix.append([math.cos(_) for _ in x])

    A = np.array(matrix).T
    coef = np.linalg.inv(A.T @ A) @ A.T @ y[:, None]

    return coef


def get_equ_val(x:np.array, coef:np.array)->np.array:

    if len(coef.shape)>1:
        coef = coef.squeeze()

    matrix = [np.ones(len(x))]
    matrix.append([math.sin(_) for _ in x])
    matrix.append([math.cos(_) for _ in x])

    matrix = np.array(matrix).T

    return matrix.dot(coef)


def main():
    x = np.array([0.1, 0.9, 1.9, 2.3, 3, 4.1, 5.2,
                 5.9, 6.8, 8.1, 8.7, 9.2, 10.1, 12])
    y = np.array([20, 24, 27, 29, 32, 37.3, 36.4,
                 32.4, 28.5, 30, 38, 43, 40, 32])

    # plot dot
    plt.scatter(x, y)

    # find best coef of my equation
    coef = find_coef(x, y)

    print(f"Coef: {coef}")
    y_pred = get_equ_val(x, coef)

    plt.plot(x, y_pred, 'r')

    # save plot to png file
    plt.savefig('./result/q4.png')

    mse = MSE(y_pred, y)
    print(f"MSE: {mse:.3f}")


if __name__ == "__main__":
    main()
