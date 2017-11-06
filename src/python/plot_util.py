import numpy as np
import matplotlib.pyplot as plt


def mesh_2d_grid(range1=[0, 5], range2=[0, 5]):
    x = np.linspace(range1[0], range1[1], 250)
    y = np.linspace(range2[0], range2[1], 250)

    xv, yv = np.meshgrid(x, y)
    return {"x": xv, "y": yv}


def get_func_value(func, mesh):
    z = np.zeros(mesh["x"].shape)
    nrow = mesh["x"].shape[0]
    ncol = mesh["x"].shape[1]
    for irow in range(nrow):
        for icol in range(ncol):
            z[irow, icol] = func([mesh["x"][irow, icol],
                                  mesh["y"][irow, icol]])

    return z


def plot_contour(mesh, z, contour_range=None):
    if contour_range is not None:
        levels = np.linspace(contour_range[0], contour_range[1], 20)
    else:
        levels = None

    CS = plt.contour(mesh["x"], mesh["y"], z, colors="black", levels=levels)
    plt.clabel(CS, inline=1, fontsize=10)
