from optimizer import Optimizer
import numpy as np


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

            # print(hess1)
            xs.append(x1)
            # update status
            niter += 1
            g0 = g1
            x0 = x1
            hess0 = hess1

        return {"x": x1, "path": xs}


class LBFGS(Optimizer):

    def __init__(self, func):
        Optimizer.__init__(self, func)

    def get_search_direction(self, s_array, y_array, gk):
        if len(s_array) != len(y_array):
            raise ValueError("sk and yk mismatch")

        q = np.copy(gk)

        m = len(s_array)
        print("Updating using last %d iterations..." % m)
        rhos = np.zeros(m)
        alphas = np.zeros(m)

        # left
        for i in range(m-1, -1, -1):
            yi = y_array[i]
            si = s_array[i]
            rhoi = 1 / np.dot(yi, si)
            rhos[i] = rhoi
            ai = rhoi * np.dot(si, q)
            alphas[i] = ai
            q = q - ai * yi

        # Hessian 0
        ndim = len(q)
        hess0 = np.zeros((ndim, ndim))
        np.fill_diagonal(hess0, 1)
        r = np.dot(hess0, q)

        # right
        for i in range(0, m, 1):
            yi = y_array[i]
            si = s_array[i]
            beta = rhos[i] * np.dot(yi, r)
            r = r + si * (alphas[i] - beta)

        return -r

    def optimize(self, starting_point, m=10, threshold=0.001, max_iter=10000,
                 verbose=False):

        x0 = np.array(starting_point, dtype=np.float)
        g0 = self.func_wrapper.gradientAt(x0)
        xs = [x0]

        y_array = []
        s_array = []
        converged = False
        niter = 0
        while(not converged and niter < max_iter):
            if niter == 0:
                # steep descent
                p0 = -g0
            else:
                p0 = self.get_search_direction(s_array, y_array, g0)

            alpha, x1 = self.line_search(x0, p0)

            if(np.linalg.norm(g0) < threshold):
                converged = True

            if verbose:
                print("[%d] alpha: %.4f | x: %s | y: %s" %
                      (niter, alpha, x1, self.func_wrapper.valueAt(x1)))

            if niter >= m:
                del y_array[0]
                del s_array[0]

            g1 = self.func_wrapper.gradientAt(x1)
            s0 = x1 - x0
            s_array.append(s0)
            y0 = g1 - g0
            y_array.append(y0)

            xs.append(x1)
            # update status
            g0 = g1
            x0 = x1
            niter += 1

        return {"x": x1, "path": xs}
