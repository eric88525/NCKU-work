import matplotlib.pyplot as plt

def draw_plt(x_list: list, y_list: list, x_range: list, y_range: list, method_name: str):
    plt.style.use('fivethirtyeight')
    plt.plot(x_list, y_list, '.')
    plt.axis(x_range + y_range)
    plt.title(f"{method_name} min: f({x_list[-1]:.3f}) = {y_list[-1]:.3f}")
    plt.show()