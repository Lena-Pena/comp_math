import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# def f():
#     return (x ** 5 + 2 * x ** 4 - 3 * x ** 3 + 4 * x ** 2 - 5) / (x + 2)


# Точное значение производной заданной функции
def tdiff(x):
    return 4 * x ** 3 - 6 * x + 10 - 35 / (x + 2) ** 2


def diff1(x_0, h, f):
    return (f(h + x_0) - f(x_0)) / h


def diff2(x_0, h, f):
    return (4 * f(x_0 + h) - 3 * f(x_0) - f(x_0 + 2 * h)) / h


class AutoDiffNumber:
    def __init__(self, a=None, b=None):
        # x = (a1, b1) = a1 + b1 * e
        # y = (a2, b2) = a2 + b2 * e
        self.a = a
        self.b = b

    def __add__(self, other):
        # (a1 + b1 * e) + (a2 + b2 * e) = (a1 + a2) +  (b1 + b2) * e
        return AutoDiffNumber(self.a + other.a, self.b + other.b)

    __radd__ = __add__

    def __mul__(self, other):
        # (a1 + b1 * e) * (a2 + b2 * e) = a1 * a2 + (b1 * a2 + a1 * b2)e
        return AutoDiffNumber(self.a * other.a, self.b * other.a + self.a * other.b)

    __rmul__ = __mul__

    def __sub__(self, other):
        # (a1 + b1 * e) - (a2 + b2 * e) = (a1 - a2) + (b1 - b2) * e
        return AutoDiffNumber(self.a - other.a, self.b - other.b)

    def __rsub__(self, other):
        return AutoDiffNumber(other.a - self.a, other.b - self.b)

    def __truediv__(self, other):
        # (a1 + b1 * e) / (a2 + b2 * e) = (a1 / a2) + (b1 * a2 - a1 * b2) * e / (a2 ** 2)
        return AutoDiffNumber(self.a / other.a, (self.b * other.a - self.a * other.b) / (other.a ** 2))

    def __rtruediv__(self, other):
        return AutoDiffNumber(other.a / self.a, (other.b * self.a - other.a * self.b) / (self.a ** 2))

    def __pow__(self, other):
        # (a1 + b1 * e) ** n = a1 ** n + b1 * n * a1 ** (n - 1) * e
        return AutoDiffNumber(self.a ** other.a, self.b * self.a ** (other.a - 1))

    def __neg__(self):
        return AutoDiffNumber(-self.a, -self.b)


def forward_autodiff(graph, x):
    return graph.forward(x).b

if __name__ == '__main__':
    x = AutoDiffNumber()
    f = (x ** 5 + 2 * x ** 4 - 3 * x ** 3 + 4 * x ** 2 - 5) / (x + 2)  # Дифференцируемая функция
    gx = np.random.uniform(low=-1, high=1, size=(100,))
    gx.sort()
    autdiff = np.array([forward_autodiff(f, i) for i in gx])
    plt.plot(gx, tdiff(gx), "r-", linewidth=2, label="$f'(x)$")
    plt.plot(gx, autdiff, "b:", linewidth=2, label="$f_a'(x)$")
    plt.gca().set_xlabel("x")
    plt.gca().set_ylabel("y")
    plt.legend()
    plt.show()


