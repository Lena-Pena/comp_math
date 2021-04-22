import numpy as np
from matplotlib import pyplot as plt
from sys import argv

x = int(argv[1])


def g(x):
    return (x ** 2) * np.sin(x)


def f(x):
    return (x ** 5 + 2 * x ** 4 - 3 * x ** 3 + 4 * x ** 2 - 5) / (x + 2)


def derivative_f(x):
    return (4 * x ** 5 + 16 * x ** 4 + 10 * x ** 3 - 14 * x ** 2 + 16 * x - 5) / ((x + 2) ** 2)


def derivative_g(x):
    return 2 * x * np.sin(x) + (x ** 2) * np.cos(x)


def diff1(x_0, h, f):
    return (f(x_0 + h) - f(x_0)) / h


def diff2(x_0, h, f):
    return (4 * f(x_0 + h) - 3 * f(x_0) - f(x_0 + 2 * h)) / (h + h)


def error_diff1(x, h):
    return np.abs(derivative_g(x) - diff1(x, h, g))


def error_diff2(x, h):
    return np.abs(derivative_g(x) - diff2(x, h, g))


fig, axes = plt.subplots(1, 1, figsize=(13, 8))
h = np.logspace(-16, 0, 100)
h_for_sq = np.logspace(-4.5, 0, 100)


diff_1_line = axes.loglog(h, error_diff1(x, h), 'o', label='diff1')
diff_2_line = axes.loglog(h, error_diff2(x, h), 'o', label='diff2')
axes.grid()
axes.set_xlabel(r'$h$', fontsize=20)
axes.set_ylabel(r'$E$', fontsize=20)
axes.loglog(h_for_sq, 10*h_for_sq, '-', label=r'$0(h)$', color='blue')
axes.loglog(h_for_sq, 10*h_for_sq**2, '-', label=r'$0(h^2)$', color='orange')
axes.tick_params(labelsize=20)

plt.legend(loc='lower left', borderaxespad=0., fontsize=20)
plt.tight_layout()
plt.show()




