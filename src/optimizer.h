#ifndef OPTIMIZER
#define OPTIMIZER

#include<iostream>
#include<vector>
#include<functional>
#include<cmath>
#include<algorithm>
#include"functionwrapper.h"
#include"utils.h"


using IterationPoint = vector<double>;
using Gradient = vector<double>;
using SearchDirection = vector<double>;


namespace DirectionUpdate {
  // steep descent update search direction
  SearchDirection sd_update(const Gradient &new_gradient);

  // conjugate update search direction in one iteration
  SearchDirection cg_update(const Gradient &last_gradient, const SearchDirection &last_direction,
                            const Gradient &new_gradient);

  // L-BFGS update search direction
  SearchDirection lbfgs_update(const vector<IterationPoint> &sks, const vector<Gradient> &yks,
                               const Gradient &new_gradient);
}

namespace FuncOptimizer {
  /*
  * Test calss for update methods
  */
  class FunctionOptimizer {
  public:
    FunctionOptimizer(FunctionWrapper _fw) : func_wrapper(_fw) {};

    virtual ~FunctionOptimizer() {}

    virtual double line_search(
        const IterationPoint &start_point, const SearchDirection &direc,
        IterationPoint &end_point, const int max_iter = 10000);

  protected:
    FunctionWrapper func_wrapper;
  };

  class SteepDescent : public FunctionOptimizer {
  public:
    SteepDescent(FunctionWrapper _fw) : FunctionOptimizer(_fw) {};

    virtual ~SteepDescent() {};

    virtual vector<IterationPoint> optimize (
        const IterationPoint &start_point, const double threshold = 0.001,
        const int max_iter = 10000, const bool verbose = false);
  };

  class ConjugateGradient : public FunctionOptimizer {
  public:
    ConjugateGradient(FunctionWrapper _fw) : FunctionOptimizer(_fw) {};

    virtual ~ConjugateGradient() {}

    virtual vector<IterationPoint> optimize (
        const IterationPoint &start_point, const double threshold = 0.001,
        const int max_iter = 10000, const bool verbose = false);
  };

  class LBFGS : public FunctionOptimizer {
  public:
    LBFGS(FunctionWrapper _fw) : FunctionOptimizer(_fw) {};

    virtual ~LBFGS() {};

    virtual vector<IterationPoint> optimize (
        const IterationPoint &start_point, const double threshold = 0.001,
        const int max_iter = 20, const bool verbose = false, const int memory=10);
  };
} // end namespace

#endif
