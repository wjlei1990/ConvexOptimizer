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
  template<typename T>
  using IterationPoint = DirectionUpdate::IterationPoint<T>;

  template<typename T>
  using Gradient = DirectionUpdate::Gradient<T>;

  template<typename T>
  using SearchDirection = DirectionUpdate::SearchDirection<T>;

  // result struct to store the optimization path
  template<typename T>
  struct OptimResult {
    OptimResult() {}
    ~OptimResult() {}
    IterationPoint<T> min_point() { return path.back(); }
    int num_iterations() { return path.size(); }

    vector<IterationPoint<T>> path;
    bool converged;
  };

  template<typename T>
  class FunctionOptimizer {
  public:
    FunctionOptimizer(FunctionWrapper<T> _fw) : func_wrapper(_fw) {};

    virtual ~FunctionOptimizer() {}

    virtual T line_search(
        const IterationPoint<T>& start_point, const SearchDirection<T>& direc,
        IterationPoint<T>& end_point, const int max_iter = 10000);

  protected:
    FunctionWrapper<T> func_wrapper;
  };

  template<typename T>
  class SteepDescent : public FunctionOptimizer<T> {
  public:
    using FunctionOptimizer<T>::func_wrapper;
    using FunctionOptimizer<T>::line_search;

    SteepDescent(FunctionWrapper<T> _fw) : FunctionOptimizer<T>(_fw) {};

    virtual ~SteepDescent() {};

    virtual OptimResult<T> optimize (
        const IterationPoint<T> start_point, const double threshold = 0.001,
        const int max_iter = 10000, const bool verbose = false);
  };

  template<typename T>
  class ConjugateGradient : public FunctionOptimizer<T> {
  public:
    using FunctionOptimizer<T>::func_wrapper;
    using FunctionOptimizer<T>::line_search;

    ConjugateGradient(FunctionWrapper<T> _fw) : FunctionOptimizer<T>(_fw) {};

    virtual ~ConjugateGradient() {}

    virtual OptimResult<T> optimize (
        const IterationPoint<T> start_point, const double threshold = 0.001,
        const int max_iter = 10000, const bool verbose = false);
  };

  template<typename T>
  class LBFGS : public FunctionOptimizer<T> {
  public:
    using FunctionOptimizer<T>::func_wrapper;
    using FunctionOptimizer<T>::line_search;

    LBFGS(FunctionWrapper<T> _fw) : FunctionOptimizer<T>(_fw) {};

    virtual ~LBFGS() {};

    virtual OptimResult<T> optimize (
        const IterationPoint<T> start_point, const double threshold = 0.001,
        const int max_iter = 10000, const bool verbose = false, const int memory=10);
  };
} // end namespace

#include "optimizer.inl"

#endif
