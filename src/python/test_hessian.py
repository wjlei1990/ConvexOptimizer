from optimizer import FuncWrapper
import numpy as np


def func(x):
    return x[0] ** 3 + x[1] ** 3


fw = FuncWrapper(func)
print fw.gradientAt([1, 1])
#print fw.hessian([0, 0])
print fw.hessian([2, 1])
print fw.inverseHessian([2, 1])

point = [2, 1]
print np.dot(fw.hessian(point), fw.inverseHessian(point))
