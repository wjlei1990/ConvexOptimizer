#ifndef DIREC_UPDATE
#define DIREC_UPDATE

#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <boost/mpi.hpp>
#include "vec_utils.hpp"

namespace mpi = boost::mpi;

namespace DirectionUpdate {

  template<typename T>
  using IterationPoint = vector<T>;

  template<typename T>
  using Gradient = vector<T>;

  template<typename T>
  using  SearchDirection = vector<T>;

  // steep descent update search direction
  template<typename T>
  SearchDirection<T> sd_update(const Gradient<T>&);

  template<typename T>
  SearchDirection<T> sd_update(const Gradient<T>&, const mpi::communicator&);

  // conjugate update search direction in one iteration
  template<typename T>
  SearchDirection<T> cg_update(const Gradient<T>& last_gradient, const  SearchDirection<T> &last_direction,
                               const Gradient<T>& new_gradient);

  template<typename T>
  SearchDirection<T> cg_update(const Gradient<T>& last_gradient, const  SearchDirection<T> &last_direction,
                               const Gradient<T>& new_gradient, const mpi::communicator& world);

  // L-BFGS update search direction
  template<typename T>
  SearchDirection<T> lbfgs_update(const vector<IterationPoint<T>>& sks, const vector<Gradient<T>>& yks,
                                  const Gradient<T>& new_gradient);

  template<typename T>
  SearchDirection<T> lbfgs_update(const vector<IterationPoint<T>>& sks, const vector<Gradient<T>>& yks,
                                  const Gradient<T>& new_gradient, const mpi::communicator& world);
}

#include "direcUpdate.inl"

#endif
