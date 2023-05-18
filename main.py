import math
from math import sqrt

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from numpy import sort

H = 0.5  # m
D = 0.1
DT = 0.1  # s
SIZE = 10
X, Y, Z = 0, 1, 2
SIM_TIME = 1200

body_cusp = ((-0.5, 0.5), (-3.2, -2), (-2, 2))  # y coordinate of body parrallel to yz

person_nostril_right = (0.1, 2, 1)
person_nostril_left = (-0.1, 2, 1)
person_mouth = (0, 2, 0.5)


def dist(loc1, loc2):
    return sqrt((loc2[X] - loc1[X]) ** 2 + (loc2[Y] - loc1[Y]) ** 2 + (loc2[Z] - loc1[Z]) ** 2)


def in_body(loc) -> bool:
    return all([body_cusp[coord][0] <= loc[coord] <= body_cusp[coord][1] for coord in range(len(body_cusp))])


def on_cusp(loc) -> (int, int):
    min_coord = -1
    min_dir = -1
    m = 10e80
    if not in_body(loc):
        return -1, -1

    for coord in range(len(body_cusp)):
        for dir in range(len(body_cusp[coord])):
            if min_coord == -1:
                min_coord = coord
                min_dir = dir
                m = abs(loc[coord] - body_cusp[coord][dir])
            elif abs(loc[coord] - body_cusp[coord][dir]) < m:
                min_coord = coord
                min_dir = dir
                m = abs(loc[coord] - body_cusp[coord][dir])

    return min_coord, min_dir


def draw_face(ax):
    x = [person_nostril_left[X], person_nostril_right[X], person_mouth[X]]
    y = [person_nostril_left[Y], person_nostril_right[Y], person_mouth[Y]]
    z = [person_nostril_left[Z], person_nostril_right[Z], person_mouth[Z]]

    ax.scatter(x, y, z, color="green")


def draw_body(ax):
    xs = []
    ys = []
    zs = []
    d = (xs, ys, zs)
    for c in range(3):
        if c == 2:
            continue
        for c1 in np.linspace(body_cusp[(c + 1) % 3][0], body_cusp[(c + 1) % 3][1], int((body_cusp[(c + 1) % 3][1] - body_cusp[(c + 1) % 3][0]) // (H / 1.5))):
            for c2 in np.linspace(body_cusp[(c + 2) % 3][0], body_cusp[(c + 2) % 3][1], int((body_cusp[(c + 2) % 3][1] - body_cusp[(c + 2) % 3][0]) // (H / 1.5))):
                d[c].append(body_cusp[c][0])
                d[c].append(body_cusp[c][1])
                d[(c + 1) % 3].append(c1)
                d[(c + 1) % 3].append(c1)
                d[(c + 2) % 3].append(c2)
                d[(c + 2) % 3].append(c2)
    ax.scatter(xs, ys, zs, color="red")


def get_bank(body_cusp, loc):
    min_dist = 1e20
    min_loc = [-1, -1, -1]
    for c in range(3):
        a1 = body_cusp[(c + 1) % 3]
        a2 = body_cusp[(c + 2) % 3]
        for c1 in range(loc_to_index(a1[0]), loc_to_index(a1[1])):
            for c2 in range(loc_to_index(a2[0]), loc_to_index(a2[1])):
                loc1 = [0, 0, 0]
                loc1[c] = body_cusp[c][0]
                loc1[(c + 1) % 3] = c1
                loc1[(c + 2) % 3] = c2

                loc2 = loc1.copy()
                loc2[c] = body_cusp[c][1]

                if dist(loc1, loc) < min_dist:
                    min_dist = dist(loc1, loc)
                    min_loc = loc1
                if dist(loc2, loc) < min_dist:
                    min_dist = dist(loc2, loc)
                    min_loc = loc2

    return min_loc


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
    try:
        return (grid[loc_1] - 2 * grid[loc] + grid[loc_2]) / (4 * H ** 2)
    except RuntimeWarning as e:
        pass


def index_to_loc(index):
    return np.array(index) * H + (-SIZE / 2)


def loc_to_index(loc):
    return int((loc + SIZE / 2) // H)


def f(grid):
    ret = D * np.array([[[dfdx2(grid, (x, y, z)) + dfdy2(grid, (x, y, z)) + dfdz2(grid, (x, y, z)) for z in range(len(grid[0][0]))] for y in range(len(grid[0]))] for x in range(len(grid))])
    for x in range(loc_to_index(body_cusp[0][0]), loc_to_index(body_cusp[0][1])):
        for y in range(loc_to_index(body_cusp[1][0]), loc_to_index(body_cusp[1][1])):
            for z in range(loc_to_index(body_cusp[2][0]), loc_to_index(body_cusp[2][1])):
                min_loc = get_bank(body_cusp, [x, y, z])
                # ret[min_loc] += ret[x, y, z] + grid[x, y, z] / DT
                ret[x, y, z] = -grid[x, y, z] / DT

    return ret


def euler(grid):
    return grid + f(grid) * DT


def generate_grid(size):
    grid = np.zeros([int(size // H), int(size // H), int(size // H)])
    grid[(0, int(size // H // 2), int(size // H // 2))] = 1
    # grid[(int(size // H) - 1, int(size // H // 2), int(size // H // 2))] = 1
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

    nostril_right_arc = []
    nostril_left_arc = []
    mouth_arc = []

    while t < SIM_TIME:
        print("\r%f.3 %c [%s%s]" % (t / SIM_TIME * 100, '%', '=' * int(t / SIM_TIME * 20), ' ' * (20 - int(t / SIM_TIME * 20))), end='')
        grid_arc.append(grid)
        t_arc.append(t)
        nostril_left_arc.append(grid[tuple([loc_to_index(c) for c in person_nostril_left])])
        nostril_right_arc.append(grid[tuple([loc_to_index(c) for c in person_nostril_right])])
        mouth_arc.append(grid[tuple([loc_to_index(c) for c in person_mouth])])
        t += DT
        grid = euler(grid)
        # update_cusp(grid)

    def animate(i):
        if i >= SIM_TIME / DT:
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
        draw_body(ax)
        draw_face(ax)

    # ani = animation.FuncAnimation(fig, animate, interval=DT)

    # plt.show()

    fig, plots = plt.subplots(2, 2)
    fig.tight_layout(pad=1.0)

    plots[0][0].scatter(t_arc, nostril_left_arc)
    plots[0][0].legend(["left_nostril"])
    plots[0][1].scatter(t_arc, nostril_right_arc)
    plots[0][1].legend(["right_nostril"])
    plots[1][0].scatter(t_arc, mouth_arc)
    plots[1][0].legend(["mouth"])
    plt.show()


if __name__ == '__main__':
    np.seterr(all='raise')
    main()
