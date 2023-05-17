from math import sqrt

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

H = 0.5  # m
D = 0.242
DT = 0.1  # s
SIZE = 10
X, Y, Z = 0, 1, 2
SIM_TIME = 10

#
# def dist(loc1, loc2):
#     return sqrt((loc2[X] - loc1[X]) ** 2 + (loc2[Y] - loc1[Y]) ** 2 + (loc2[Z] - loc1[Z]) ** 2)


def dfdx2(grid: np.ndarray, loc: (int, int)):
    loc_1 = (min(len(grid) - 1, loc[X] + 2), loc[Y], loc[Z])
    loc_2 = (max(0, loc[X] - 2), loc[Y], loc[Z])
    return (grid[loc_1] - 2 * grid[loc] + grid[loc_2]) / (4 * H ** 2)


def dfdy2(grid: np.ndarray, loc: (int, int)):
    loc_1 = (loc[X], min(len(grid[0]) - 1, loc[Y] + 2), loc[Z])
    loc_2 = (loc[X], max(0, loc[Y] - 2), loc[Z])
    return (grid[loc_1] - 2 * grid[loc] + grid[loc_2]) / (4 * H ** 2)


def dfdz2(grid: np.ndarray, loc: (int, int)):
    loc_1 = (loc[X], loc[Y], min(len(grid[0][0]) - 1, loc[Z] + 2))
    loc_2 = (loc[X], loc[Y], max(0, loc[Z] - 2))
    return (grid[loc_1] - 2 * grid[loc] + grid[loc_2]) / (4 * H ** 2)


def index_to_loc(loc):
    return np.array(loc) * H + (-SIZE / 2)


def f(grid):
    return D * np.array([[[dfdx2(grid, (x, y, z)) + dfdy2(grid, (x, y, z)) + dfdz2(grid, (x, y, z)) for z in range(len(grid[0][0]))] for y in range(len(grid[0]))] for x in range(len(grid))])


def euler(grid):
    return grid + f(grid) * DT


def generate_grid(size):
    grid = np.zeros([int(size // H), int(size // H), int(size // H)])
    grid[(0, int(size // H // 2), int(size // H // 2))] = 1
    grid[(int(size // H)-1, int(size // H // 2), int(size // H // 2))] = 1
    return grid


def main():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    grid = np.array(generate_grid(SIZE))

    grid_arc = []
    t_arc = []

    t = 0
    style.use('fivethirtyeight')
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    while t < SIM_TIME:
        grid_arc.append(grid)
        t_arc.append(t)
        t += DT
        grid = euler(grid)

    def animate(i):
        if i >= SIM_TIME/DT:
            ani.pause()
            return
        xs = []
        ys = []
        zs = []
        cs = []
        for x in range(len(grid)):
            for y in range(len(grid[x])):
                for z in range(len(grid[x][y])):
                    xs.append(x * H - SIZE / 2)
                    ys.append(y * H - SIZE / 2)
                    zs.append(z * H - SIZE / 2)
                    cs.append(grid_arc[i][(x, y, z)])

        ax.clear()
        ax.scatter(xs, ys, zs, s=np.array(cs) * 100)

    ani = animation.FuncAnimation(fig, animate, interval=DT)

    plt.show()


if __name__ == '__main__':
    main()
