from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt


class FuncWrapper(object):

    def __init__(self, func):
        self.func = func

    def valueAt(self, x):
        return self.func(x)

    def finite_diff_grad(self, x, i, dx, order="second"):
        x0 = np.copy(x)
        if order == "second":
            x0[i] -= dx

        x1 = np.copy(x)
        x1[i] += dx

        v0 = self.valueAt(x0)
        v1 = self.valueAt(x1)
        if order == "second":
            ret = (v1 - v0) / (2 * dx)
        else:
            ret = (v1 - v0) / dx

        return ret

    def gradientAt(self, x):
        x0 = np.array(x, dtype=np.float)
        # v0 = self.valueAt(x0)

        dx = 0.001
        grad = np.zeros(x0.shape)
        for i in range(len(x0)):
            # x1 = np.copy(x0)
            # x1[i] += dx
            # v1 = self.valueAt(x1)
            # grad[i] = (v1 - v0) / dx
            grad[i] = self.finite_diff_grad(x0, i, dx)

        return grad

    def hessian(self, x):
        x0 = np.array(x, dtype=np.float)
        dx = 0.001

        nx = len(x)
        hess = np.zeros([nx, nx])
        for i in range(nx):
            for j in range(i, nx):
                x1 = np.copy(x0)
                x1[j] += dx
                g1 = self.finite_diff_grad(x1, i, dx)
                g0 = self.finite_diff_grad(x0, i, dx)
                deriv = (g1 - g0) / dx
                hess[i][j] = deriv

        for i in range(nx):
            for j in range(0, i):
                hess[i][j] = hess[j][i]

        return hess

    def inverseHessian(self, x):
        hessian = self.hessian(x)
        return np.linalg.inv(hessian)


class Optimizer(object):
    def __init__(self, func):
        self.func_wrapper = FuncWrapper(func)

    def optimize(self):
        pass

    def line_search(self, x0, p, max_iter=10000, dalpha=0.0001):
        v0 = self.func_wrapper.valueAt(x0)
        find = False
        niter = 0

        iters = [0]
        vs = [v0]
        while (not find and niter < max_iter):
            x1 = x0 + dalpha * p
            v1 = self.func_wrapper.valueAt(x1)
            vs.append(v1)
            #print("v0 and v1: ", v0, v1)
            #print(x0, v0, x1, v1)
            if(v1 > v0):
                find = True
            x0 = x1
            v0 = v1
            niter += 1
            iters.append(niter)

        #plt.figure()
        #plt.plot(iters, vs)
        #plt.show()

        return dalpha*niter, x0

    def line_search_fix(self, x0, p, alpha=0.05):
        return alpha, x0 + alpha * p


class SteepDescent(Optimizer):
    def __init__(self, func):
        Optimizer.__init__(self, func)

    def optimize(self, starting_point, threshold=0.001, max_iter=100000,
                 verbose=False):
        converged = False
        x0 = np.array(starting_point, dtype=np.float64)
        g0 = self.func_wrapper.gradientAt(x0)

        xs = [x0]
        vs = [self.func_wrapper.valueAt(x0)]
        alphas = []

        niter = 0
        while (not converged and niter < max_iter):
            alpha, x1 = self.line_search(x0, -g0)
            g1 = self.func_wrapper.gradientAt(x1)
            # check converge
            if(np.linalg.norm(g1) < threshold):
                converged = True

            #print("grad:", grad)
            #print("alpha:", alpha)
            #print("x1:", x1)

            orth = np.dot(g1, g0) / np.dot(g1, g1)
            # keep the record
            xs.append(x1)
            alphas.append(alpha)
            v1 = self.func_wrapper.valueAt(x1)
            vs.append(v1)
            diff = v1 - vs[niter]
            if verbose:
                print("[%3d] diff: %10.4f | alpha: %8.4f | orth: %8.4f | " %
                      (niter, diff, alpha, orth) + " | x: %s" % (x1) +
                      " | y: %.4f" % self.func_wrapper.valueAt(x1))

            x0 = x1
            g0 = g1
            niter += 1

        return {"x": x1, "path": xs}


class ConjugateGradient(Optimizer):
    def __init__(self, func):
        Optimizer.__init__(self, func)

    def optimize(self, starting_point, threshold=0.001, max_iter=100000,
                 orth_threshold=0.1, verbose=False):

        converged = False
        x0 = np.array(starting_point, dtype=np.float64)
        # f0 = self.func_wrapper.valueAt(x0)
        g0 = self.func_wrapper.gradientAt(x0)
        p0 = -g0

        xs = [x0]
        # alphas = []

        niter = 0
        while(not converged and niter < max_iter):
            alpha, x1 = self.line_search(x0, p0)
            g1 = self.func_wrapper.gradientAt(x1)

            # check for restart
            orth = np.dot(g1, g0) / np.dot(g1, g1)
            beta = np.dot(g1, g1) / np.dot(g0, g0)
            if np.abs(orth) > 0.1:
                restart = True
                p1 = -g1
            else:
                restart = False
                p1 = -g1 + beta * p0

            if(np.sum(np.abs(g1)) < threshold):
                converged = True

            if verbose:
                v1 = self.func_wrapper.valueAt(x1)
                diff = v1 - self.func_wrapper.valueAt(x0)
                print("[%3d] diff: %10.4e | alpha: %.2f "
                      % (niter, diff, alpha)
                      + "|  orth: %.4f | beta: %10.4f |" % (beta, orth)
                      + " Restart: %s" % restart
                      + " | y: %.4f" % self.func_wrapper.valueAt(x1))

            xs.append(x1)
            # update
            p0 = p1
            g0 = g1
            x0 = x1

            niter += 1

        return {"x": x1, "path": xs}
