import math
from math import sqrt

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from numpy import sort
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


H = 0.2# m
D = 0.4
DT = 0.05  # s
SIZE = 3
X, Y, Z = 0, 1, 2
SIM_TIME = 60

body_cusp = ((-0.5, 0.5), (-1, -0.5), (-1.5, 1.5))  # y coordinate of body parrallel to yz
body2 = ((-0.5, 0.5), (0.5, 1), (-1.5, 1.5))
spread_loc = (0, -0.4, 0.5)

person_nostril_right = (0.1, 0.4, 1.1)
person_nostril_left = (-0.1, 0.4, 1.1)
person_mouth = (0, 0.4, 0.5)

BOX_X_LENGTH, BOX_Y_LENGTH, BOX_Z_LENGTH = 1, 0.5, 3
BOX_X, BOX_Y, BOX_Z = -0.5, 0.5, 1.5
# STARTING_SELIVA_INDEX = (4, int(SIZE // H // 2), int(SIZE // H // 2))


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

def plot_box(x, y, z, x_length, y_length, z_length, ax, color):
    x_bottom_array = [x, x, x + x_length, x + x_length]
    y_bottom_array = [y, y + y_length, y + y_length, y]
    z_bottom_array = [z - z_length, z - z_length, z - z_length, z - z_length]

    x_top_array = [x, x, x + x_length, x + x_length]
    y_top_array = [y, y + y_length, y + y_length, y]
    z_top_array = [z, z, z, z]

    x_side1_array = [x, x, x, x]
    y_side1_array = [y, y + y_length, y + y_length, y]
    z_side1_array = [z, z, z - z_length, z - z_length]

    x_side2_array = [x, x, x + x_length, x + x_length]
    y_side2_array = [y, y, y, y]
    z_side2_array = [z, z - z_length, z - z_length, z]

    x_side3_array = [x + x_length, x + x_length, x + x_length, x + x_length]
    y_side3_array = [y, y + y_length, y + y_length, y]
    z_side3_array = [z, z, z - z_length, z - z_length]

    x_side4_array = [x + x_length, x, x, x + x_length]
    y_side4_array = [y + y_length, y + y_length, y + y_length, y + y_length]
    z_side4_array = [z, z, z - z_length, z - z_length]

    # Define the vertices for the polygons
    vertices_bottom = [
        list(zip(x_bottom_array, y_bottom_array, z_bottom_array))]
    vertices_top = [list(zip(x_top_array, y_top_array, z_top_array))]
    vertices_side1 = [list(zip(x_side1_array, y_side1_array, z_side1_array))]
    vertices_side2 = [list(zip(x_side2_array, y_side2_array, z_side2_array))]
    vertices_side3 = [list(zip(x_side3_array, y_side3_array, z_side3_array))]
    vertices_side4 = [list(zip(x_side4_array, y_side4_array, z_side4_array))]

    # Create Poly3DCollection objects for each face
    poly_bottom = Poly3DCollection(vertices_bottom, alpha=0.8, facecolor=color)
    poly_top = Poly3DCollection(vertices_top, alpha=0.8, facecolor=color)
    poly_side1 = Poly3DCollection(vertices_side1, alpha=0.8, facecolor=color)
    poly_side2 = Poly3DCollection(vertices_side2, alpha=0.8, facecolor=color)
    poly_side3 = Poly3DCollection(vertices_side3, alpha=0.8, facecolor=color)
    poly_side4 = Poly3DCollection(vertices_side4, alpha=0.8, facecolor=color)

    # Add the polygons to the plot
    ax.add_collection3d(poly_bottom)
    ax.add_collection3d(poly_top)
    ax.add_collection3d(poly_side1)
    ax.add_collection3d(poly_side2)
    ax.add_collection3d(poly_side3)
    ax.add_collection3d(poly_side4)

def draw_body(ax):
    xs = []
    ys = []
    zs = []
    d = (xs, ys, zs)
    xs2 = []
    ys2 = []
    zs2 = []
    d2 = (xs2, ys2, zs2)
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
        for c1 in np.linspace(body2[(c + 1) % 3][0], body2[(c + 1) % 3][1], int((body2[(c + 1) % 3][1] - body2[(c + 1) % 3][0]) // (H / 1.5))):
            for c2 in np.linspace(body2[(c + 2) % 3][0], body2[(c + 2) % 3][1], int((body2[(c + 2) % 3][1] - body2[(c + 2) % 3][0]) // (H / 1.5))):
                d2[c].append(body2[c][0])
                d2[c].append(body2[c][1])
                d2[(c + 1) % 3].append(c1)
                d2[(c + 1) % 3].append(c1)
                d2[(c + 2) % 3].append(c2)
                d2[(c + 2) % 3].append(c2)
    ax.scatter(xs, ys, zs, color="red")
    ax.scatter(xs2, ys2, zs2, color="cyan")

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

def loc3_to_index(loc):
    return [loc_to_index(i) for i in loc]

def f(grid):
    ret = D * np.array([[[dfdx2(grid, (x, y, z)) + dfdy2(grid, (x, y, z)) + dfdz2(grid, (x, y, z)) for z in range(len(grid[0][0]))] for y in range(len(grid[0]))] for x in range(len(grid))])
    # ret[loc_to_index(body_cusp[0][0]):loc_to_index(body_cusp[0][1]), loc_to_index(body_cusp[1][0]):loc_to_index(body_cusp[1][1]), loc_to_index(body_cusp[2][0]):loc_to_index(body_cusp[2][1])] = \
    #     -grid[loc_to_index(body_cusp[0][0]):loc_to_index(body_cusp[0][1]), loc_to_index(body_cusp[1][0]):loc_to_index(body_cusp[1][1]), loc_to_index(body_cusp[2][0]):loc_to_index(body_cusp[2][1])] / DT
    # ret[loc_to_index(body2[0][0]):loc_to_index(body2[0][1]), loc_to_index(body2[1][0]):loc_to_index(body2[1][1]), loc_to_index(body2[2][0]):loc_to_index(body2[2][1])] = \
    #     -grid[loc_to_index(body2[0][0]):loc_to_index(body2[0][1]), loc_to_index(body2[1][0]):loc_to_index(body2[1][1]), loc_to_index(body2[2][0]):loc_to_index(body2[2][1])] / DT

    return ret
def rk4(grid):
    k1 = f(grid) * DT
    k2 = f(grid + k1 / 2) * DT
    k3 = f(grid + k2 / 2) * DT
    k4 = f(grid + k3) * DT
    return grid + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def euler(grid):

    return grid + f(grid) * DT

def generate_grid(size):
    grid = np.zeros([int(size // H), int(size // H), int(size // H)])
    index = loc3_to_index(spread_loc)
    grid[index[0], index[1], index[2]] = 5
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
    index = 0
    startin_index = loc3_to_index(spread_loc)
    while t < SIM_TIME:
        if index%50 == 0:
            grid[startin_index[0], startin_index[1], startin_index[2]] = 5
        print("\r%f.3 %c [%s%s]" % (t / SIM_TIME * 100, '%', '=' * int(t / SIM_TIME * 20), ' ' * (20 - int(t / SIM_TIME * 20))), end='')
        grid_arc.append(grid)
        t_arc.append(t)
        nostril_left_arc.append(grid[tuple([loc_to_index(c) for c in person_nostril_left])])
        nostril_right_arc.append(grid[tuple([loc_to_index(c) for c in person_nostril_right])])
        mouth_arc.append(grid[tuple([loc_to_index(c) for c in person_mouth])])
        t += DT
        grid = rk4(grid)
        # update_cusp(grid)
        index += 1

    def animate(i):
        if i >= SIM_TIME / DT:
            ani.pause()
            return
        xs = []
        ys = []
        zs = []
        cs = []
        for x in range(len(grid_arc[0])):
            for y in range(len(grid_arc[0][x])):
                for z in range(len(grid_arc[0][x][y])):
                    xs.append(x * H - SIZE / 2)
                    ys.append(y * H - SIZE / 2)
                    zs.append(z * H - SIZE / 2)
                    cs.append(abs(grid_arc[i][(x, y, z)]))

        ax.clear()
        ax.scatter(xs, ys, zs, s=np.array(cs) * 100)
        # draw_body(ax)
        draw_face(ax)
        plot_box(BOX_X, BOX_Y, BOX_Z, BOX_X_LENGTH, BOX_Y_LENGTH, BOX_Z_LENGTH,
                 ax, "red")  # body1
        plot_box(BOX_X, BOX_Y-1.5, BOX_Z, BOX_X_LENGTH, BOX_Y_LENGTH, BOX_Z_LENGTH,
                 ax, "red")  # body2
        # plot_box(BOX_X, BOX_Y + BOX_Y_LENGTH / 4, 2, BOX_X_LENGTH,
        #          BOX_Y_LENGTH / 2, 2, ax, "peachpuff")  # head
        # plot_box(BOX_X, BOX_Y - 3, BOX_Z, BOX_X_LENGTH / 2, 3, 1, ax,
        #          "red")  # left hand
        # plot_box(BOX_X, BOX_Y + BOX_Y_LENGTH, BOX_Z, BOX_X_LENGTH / 2, 3, 1, ax,
        #          "red")  # right hand
    #
    # print(nostril_left_arc)
    # print(nostril_right_arc)
    # print(mouth_arc)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ani = animation.FuncAnimation(fig, animate, interval=DT * 1000)

    plt.show()

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