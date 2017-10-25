from optimizer import SteepDescent, ConjugateGradient
from newton import NewtonMethod, BFGS
from QuasiNewton import LBFGS
from plot_util import mesh_2d_grid, get_func_value, plot_contour
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def quadratic(x):
    v = x[0] ** 2 + 2 * x[1] ** 2 + 2 * x[0] * x[1] + 2 * x[1]
    return v


def rosenbrock(x, a=1, b=100):
    v = (a - x[0]) ** 2 + b * (x[1] - x[0]**2) ** 2
    return v


def plot_func(func, optim_results, bounds=None):

    if bounds is None:
        point = optim_results[optim_results.keys()[0]]["x"]
        bounds = [[point[0]-2, point[0]+2], [point[1]-2, point[1]+2]]

    mesh = mesh_2d_grid(bounds[0], bounds[1])
    z = get_func_value(func, mesh)

    #plt.pcolormesh(mesh["x"], mesh["y"], z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(mesh["x"], mesh["y"], z, cmap=cm.rainbow, linewidth=0)

    #plot_contour(mesh, z, contour_range=[np.min(z)+0.1, 2])
    plt.figure()
    plot_contour(mesh, z)

    for key, info in optim_results.iteritems():
        x1 = [x[0] for x in info["path"]]
        x2 = [x[1] for x in info["path"]]
        if key == "bfgs":
            plt.plot(x1, x2, "--", label=key)
        else:
            plt.plot(x1, x2, "*", label=key)

    plt.legend()
    plt.show()


def main(func, starting_point=[0.0, 0.0]):

    #op = SteepDescent(func)
    #op = ConjugateGradient(func)
    #op = NewtonMethod(func)
    optim_results = {}

    op = LBFGS(func)
    bfgs_results = op.optimize(starting_point, threshold=0.0001, verbose=True)
    optim_results["lbfgs"] = bfgs_results
    #print("Minimal point: %s" % optim_results["x"])

    #op = SteepDescent(func)
    #sd_results = op.optimize(starting_point, threshold=0.0001, verbose=True)
    #optim_results["sd_results"] = sd_results

    plot_func(func, optim_results, bounds=[[-2, 2], [-1, 3]])
    #plot_func(func, optim_results)


if __name__ == "__main__":
    #main(quadratic)
    main(rosenbrock, starting_point=[-1.2, 1.0])
