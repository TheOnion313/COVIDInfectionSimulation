from math import sqrt

import numpy as np
import matplotlib

# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FFMpegWriter

H = 0.5  # m
D = 0.5
DT = 0.1  # s
SIZE = 10
X, Y, Z = 0, 1, 2
SIM_TIME = 30
BOX_X_LENGTH, BOX_Y_LENGTH, BOX_Z_LENGTH = 1.5, 2, 5
BOX_X, BOX_Y, BOX_Z = -5, -BOX_Y_LENGTH / 2, 0
STARTING_SELIVA_INDEX = (4, int(SIZE // H // 2), int(SIZE // H // 2))

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


def f(grid):
    return D * np.array([[[dfdx2(grid, (x, y, z)) + dfdy2(grid,
                                                          (x, y, z)) + dfdz2(
        grid, (x, y, z)) for z in range(len(grid[0][0]))] for y in
                          range(len(grid[0]))] for x in range(len(grid))])


def euler(grid):
    return grid + f(grid) * DT


def generate_grid(size):
    grid = np.zeros([int(size // H), int(size // H), int(size // H)])
    grid[STARTING_SELIVA_INDEX] = 1
    # grid[(int(size // H)-1, int(size // H // 2), int(size // H // 2))] = 1
    return grid


def main():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    grid = np.array(generate_grid(SIZE))
    grid_arc = []
    t_arc = []
    t = 0
    index = 0
    style.use('fivethirtyeight')
    while t < SIM_TIME:
        if index%10 == 0:
            grid[STARTING_SELIVA_INDEX] = 1
        grid_arc.append(grid)
        t_arc.append(t)
        t += DT
        grid = euler(grid)

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
        plot_box(BOX_X, BOX_Y, BOX_Z, BOX_X_LENGTH, BOX_Y_LENGTH, BOX_Z_LENGTH,
                 ax, "red")  # body
        plot_box(BOX_X, BOX_Y + BOX_Y_LENGTH / 4, 2, BOX_X_LENGTH,
                 BOX_Y_LENGTH / 2, 2, ax, "peachpuff")  # head
        plot_box(BOX_X, BOX_Y - 3, BOX_Z, BOX_X_LENGTH / 2, 3, 1, ax,
                 "red")  # left hand
        plot_box(BOX_X, BOX_Y + BOX_Y_LENGTH, BOX_Z, BOX_X_LENGTH / 2, 3, 1, ax,
                 "red")  # right hand

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 5)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    ani = animation.FuncAnimation(fig, animate, interval=DT*1000)
    # output_file = 'animation.mp4'
    # writer = FFMpegWriter(fps=30)
    # ani.save(output_file, writer=writer)
    plt.show()

if __name__ == '__main__':
    main()
