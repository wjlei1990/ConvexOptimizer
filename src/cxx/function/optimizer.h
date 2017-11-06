#ifndef OPTIMIZER
#define OPTIMIZER

#include<iostream>
#include<vector>
#include<functional>
#include<cmath>
#include<algorithm>
#include <boost/mpi.hpp>
#include "vec_utils.hpp"
#include "functionwrapper.h"
#include "direcUpdate.h"

namespace mpi = boost::mpi;

namespace FuncOptimizer {
  /*
  * Test calss for update methods
  */
  using IterationPoint = DirectionUpdate::IterationPoint;
  using Gradient = DirectionUpdate::Gradient;
  using SearchDirection = DirectionUpdate::SearchDirection;

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
