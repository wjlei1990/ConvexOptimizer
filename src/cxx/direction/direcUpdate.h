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
  using IterationPoint = vector<double>;
  using Gradient = vector<double>;
  using SearchDirection = vector<double>;

  // steep descent update search direction
  SearchDirection sd_update(const Gradient &);
  SearchDirection sd_update(const Gradient &, const mpi::communicator&);

  // conjugate update search direction in one iteration
  SearchDirection cg_update(const Gradient &last_gradient, const SearchDirection &last_direction,
                            const Gradient &new_gradient);
  SearchDirection cg_update(const Gradient &last_gradient, const SearchDirection &last_direction,
                            const Gradient &new_gradient, const mpi::communicator& world);

  // L-BFGS update search direction
  SearchDirection lbfgs_update(const vector<IterationPoint> &sks, const vector<Gradient> &yks,
                               const Gradient &new_gradient);
  SearchDirection lbfgs_update(const vector<IterationPoint> &sks, const vector<Gradient> &yks,
                               const Gradient &new_gradient, const mpi::communicator& world);
}

#endif
