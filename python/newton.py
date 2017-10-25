from optimizer import Optimizer
import numpy as np


class NewtonMethod(Optimizer):

    def __init__(self, func):
        Optimizer.__init__(self, func)

    def optimize(self, starting_point, threshold=0.001, max_iter=100000,
                 verbose=False):

        converged = False

        x0 = np.array(starting_point, dtype=np.float)
        xs = [x0]

        niter = 0
        while (not converged and niter < max_iter):
            g0 = self.func_wrapper.gradientAt(x0)
            invHess = self.func_wrapper.inverseHessian(x0)
            p0 = - np.dot(invHess, g0)
            alpha, x1 = self.line_search(x0, p0)

            if(np.linalg.norm(g0) < threshold):
                converged = True

            if verbose:
                print("[%d] alpha: %.2f | x: %s" % (niter, alpha, x1))

            xs.append(x1)
            x0 = x1
            niter += 1

        return {"x": x1, "path": xs}


class BFGS(Optimizer):
    """
    Quasi-Newton Method
    """
    def __init__(self, func):
        Optimizer.__init__(self, func)

    def optimize(self, starting_point, threshold=0.001, max_iter=10000,
                 verbose=False):

        x0 = np.array(starting_point, dtype=np.float)
        g0 = self.func_wrapper.gradientAt(x0)
        xs = [x0]

        ndim = len(g0)
        identity = np.zeros((ndim, ndim))
        np.fill_diagonal(identity, 1)
        hess0 = identity.copy()

        converged = False
        niter = 0
        while(not converged and niter < max_iter):
            print("-------------------------")
            print("g0:", g0)
            print("hess:")
            print(hess0)
            p0 = -np.dot(hess0, g0)
            print("p0:", -p0)
            alpha, x1 = self.line_search(x0, p0)

            if(np.linalg.norm(g0) < threshold):
                converged = True

            if verbose:
                print("[%d] alpha: %.4f | x: %s | y: %s" %
                      (niter, alpha, x1, self.func_wrapper.valueAt(x1)))

            s0 = x1 - x0
            g1 = self.func_wrapper.gradientAt(x1)
            y0 = g1 - g0
            rho = 1 / np.dot(y0, s0)
            print("rho:", rho)
            m1 = (identity - rho * s0.reshape(ndim, 1) * y0.reshape(1, ndim))
            m2 = (identity - rho * y0.reshape(ndim, 1) * s0.reshape(1, ndim))
            m3 = rho * s0.reshape(ndim, 1) * s0.reshape(1, ndim)
            hess1 = np.dot(np.dot(m1, hess0), m2) + m3

            #print(hess1)
            xs.append(x1)
            # update status
            g0 = g1
            x0 = x1
            hess0 = hess1
            niter += 1

        return {"x": x1, "path": xs}
