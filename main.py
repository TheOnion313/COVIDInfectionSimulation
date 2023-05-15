from math import sqrt

import numpy as np
from matplotlib import pyplot as plt

import matplotlib.animation as animation
from matplotlib import style

H = 1  # m
D = 50
DT = 0.001  # s


def dist(loc1, loc2):
    return sqrt((loc2[0] - loc1[0]) ** 2 + (loc2[1] - loc1[1]) ** 2)


def dfdx2(grid: np.ndarray, loc: (int, int)):
    loc_1 = (min(len(grid) - 1, loc[0] + 1), loc[1])
    loc_2 = (max(0, loc[0] - 1), loc[1])
    return (grid[loc_1] - 2 * grid[loc] + grid[loc_2]) / H ** 2


def dfdy2(grid: np.ndarray, loc: (int, int)):
    loc_1 = (loc[0], min(len(grid[0]) - 1, loc[1] + 1))
    loc_2 = (loc[0], max(0, loc[1] - 1))
    return (grid[loc_1] - 2 * grid[loc] + grid[loc_2]) / H ** 2


def f(grid):
    return D * np.array([[dfdx2(grid, (x, y)) + dfdy2(grid, (x, y)) for y in range(len(grid[0]))] for x in range(len(grid))])


def euler(grid):
    return grid + f(grid) * DT


def generate_grid(size):
    return [[1 / (dist((0, 0), (x - size / 2, y - size / 2)) + 0.0001) for y in np.linspace(0, size, int(size // H))]
            for x in np.linspace(0, size, int(size // H))]


def main():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    size = 10
    grid = np.array(generate_grid(size))
    print(grid)

    grid_arc = []
    t_arc = []

    t = 0
    style.use('fivethirtyeight')
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    while t < 1:
        grid_arc.append(grid)
        t_arc.append(t)
        t += DT
        grid = euler(grid)

    def animate(i):
        if i >= len(grid_arc):
            ani.pause()
            return
        xs = []
        ys = []
        zs = []
        for x in range(len(grid)):
            for y in range(len(grid[x])):
                xs.append(x * H - size / 2)
                ys.append(y * H - size / 2)
                zs.append(grid_arc[i][(x, y)])
        ax.clear()
        ax.scatter(xs, ys, zs)

    ani = animation.FuncAnimation(fig, animate, interval=DT)

    plt.show()


if __name__ == '__main__':
    main()
