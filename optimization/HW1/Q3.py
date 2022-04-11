from methods.n_dim import powell_method
import matplotlib.pyplot as plt


def himmelblau_func(x1: float, x2: float) -> float:
    return (x1*x1+x2-11)**2 + (x1+x2*x2 - 7)**2


def main():
    x_list, y_list = powell_method([-5, 5], [-5, 5], himmelblau_func)

    for i in range(len(x_list)):
        print(
            f"x1: {x_list[i][0]:.3f} x2: {x_list[i][1]:.3f} f(x1, x2): {y_list[i]:.6f}")

    fig, ax = plt.subplots()
    ax.plot([x[0] for x in x_list], [x[1]
            for x in x_list], '-o', ms=5, lw=2, alpha=0.7, mfc='orange')
    ax.grid()
    plt.style.use('fivethirtyeight')
    plt.axis([-5, 5, -5, 5])
    plt.text(x_list[0][0], x_list[0][1], "start")
    plt.text(x_list[-1][0], x_list[-1][1], "end")
    plt.title(
        f"Powell: f({x_list[-1][0]:.3f}, {x_list[-1][1]:.3f}) = {himmelblau_func(x_list[-1][0], x_list[-1][1]):.3f}")
    plt.savefig('Powell.png')
    plt.show()


if __name__ == "__main__":
    main()
