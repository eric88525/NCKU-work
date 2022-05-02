# HW2
by <`Q56104076 陳哲緯`>
# Q1

![](./result/q1.png)

### Quadratic

$$
\begin{pmatrix}
21.262 \\
3.566 \\
-0.206 \\
\end{pmatrix}
$$

### Linear

$$
\begin{pmatrix}
25.584 \\
1.167  \\
\end{pmatrix}
$$

```python
def main():
    x= np.array([0.1, 0.9, 1.9, 2.3, 3, 4.1, 5.2, 5.9, 6.8, 8.1, 8.7, 9.2, 10.1,12])
    y= np.array([20, 24, 27, 29, 32, 37.3, 36.4, 32.4, 28.5, 30, 38, 43, 40, 32])

    # get coefficients
    coeffs_2d = np.polyfit(x, y, 2)
    coeffs_1d = np.polyfit(x, y, 1)
    print(f"Quadratic: {coeffs_2d}")
    print(f"Linear: {coeffs_1d}")
    
    # plot equations 
    plt.scatter(x, y)

    # equations
    polynomial = np.poly1d(coeffs_2d)
    linear = np.poly1d(coeffs_1d)

    _x = np.linspace(x.min(), x.max(), 100)
    polynomial_y = [ polynomial(i) for i in _x ]
    linear_y = [ linear(i) for i in _x ]

    plt.plot(_x, polynomial_y, 'r-' )
    plt.plot(_x, linear_y, 'g-', )

    # save plot to png file
    plt.savefig('./result/q1.png')

```

# Q2
# Q3
# Q4
# Q5