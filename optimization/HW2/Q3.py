from cmath import inf
import numpy as np

# mean squared error of y_pred and y_label
def MSE(y_pred: np.array, y_label: np.array)->float:
    return ((y_pred-y_label)**2).sum()/len(y_label)


def main():

    x = np.array([0.1, 0.9, 1.9, 2.3, 3, 4.1, 5.2,
                 5.9, 6.8, 8.1, 8.7, 9.2, 10.1, 12])
    y = np.array([20, 24, 27, 29, 32, 37.3, 36.4,
                 32.4, 28.5, 30, 38, 43, 40, 32])

    min_mse, min_degree = inf, 1

    for i in range(1, 20):

        coeff = np.polyfit(x, y, i)
        polynomial = np.poly1d(coeff)

        y_pred = [polynomial(_x) for _x in x]
        mse = MSE(y_pred, y)

        print(f"Degree {i:>2}: MSE = {mse:.3f}")

        if mse < min_mse:
            min_mse, min_degree = mse, i

        if min_mse < 1e-3:
            break

    print(f"Best degree is {min_degree:>2}, MSE: {min_mse:.3f}")


if __name__ == "__main__":
    main()
