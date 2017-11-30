from optimizer import SteepDescent, ConjugateGradient
from newton import NewtonMethod, BFGS
from QuasiNewton import LBFGS
from plot_util import mesh_2d_grid, get_func_value, plot_contour
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import animation
from matplotlib import cm


def quadratic(x):
    v = x[0] ** 2 + 2 * x[1] ** 2 + 2 * x[0] * x[1] + 2 * x[1]
    return v


def rosenbrock(x, a=1, b=100):
    dim = len(x)
    vsum = 0
    for i in range(dim-1):
        vsum += (a - x[i]) ** 2 + b * (x[i+1] - x[i]**2) ** 2
    return vsum


def plot_traj(func, path):

    xmin, xmax, xstep = -3, 3, 0.05
    ymin, ymax, ystep = -2, 4, 0.05
    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep),
                       np.arange(ymin, ymax + ystep, ystep))
    z = get_func_value(func, {"x": x, "y": y})

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contour(x, y, z, levels=np.logspace(0, 5, 35),
               norm=LogNorm(), cmap=plt.cm.jet)
    ax.plot(1, 1, 'r*', markersize=18)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    #line, = ax.plot([], [], 'b', label='Newton-CG', lw=2)
    line, = ax.plot([], [], 'r', animated=True, lw=2)
    point, = ax.plot([], [], 'ro')
    xdata, ydata = [], []

    def init():
        line.set_data([], [])
        point.set_data([], [])
        #ax.set_xlim((xmin, xmax))
        #ax.set_ylim((ymin, ymax))
        return line, point

    def animate(i):
        print(path[i])
        xdata.append(path[i, 0])
        ydata.append(path[i, 1])
        #print(i)
        #xdata.append(0.01*i)
        #ydata.append(np.sin(i))
        line.set_data(xdata, ydata)
        point.set_data(path[i, 0], path[i, 1])
        return line, point

    #ax.legend(loc='upper left')
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(path), blit=True)
    anim.save("movie.mp4")
    #anim.to_html5_video()
    #plt.show()


def main(func, starting_point=[0.0, 0.0]):
    optim_results = {}

    op = LBFGS(func)
    lbfgs_results = op.optimize(starting_point, threshold=0.0001,
                                verbose=False)
    optim_results["lbfgs"] = lbfgs_results
    path = np.array(lbfgs_results["path"])
    print("Number of iterations to minimum:", len(lbfgs_results["path"]))
    plot_traj(func, path)
    #print("Minimal point: %s" % optim_results["x"])

    #op = SteepDescent(func)
    #sd_results = op.optimize(starting_point, threshold=0.0001, verbose=True)
    #optim_results["sd_results"] = sd_results

    #plot_func(func, optim_results, bounds=[[-2, 2], [-1, 3]])
    #plot_func(func, optim_results)


if __name__ == "__main__":
    #main(quadratic)
    main(rosenbrock, starting_point=[-1.5, 0.0])
