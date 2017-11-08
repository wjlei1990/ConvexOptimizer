#include "direcUpdate.h"

namespace mpi = boost::mpi;

namespace DirectionUpdate {

  // steep descent update search direction
  SearchDirection sd_update(const Gradient &g) {
    return -1.0 * g;
  }

  // Parallel steep descent
  SearchDirection sd_update(const Gradient& g, const mpi::communicator& world){
    // dummpy arguments in worlds since steep descent doesn't require any
    // communications between process
    return -1.0 * g;
  }

  // conjugate update search direction in one iteration
  SearchDirection cg_update(const Gradient &last_gradient, const SearchDirection &last_direction,
                            const Gradient &this_gradient)
  {
    double beta = dot(this_gradient, this_gradient) / dot(last_gradient, last_gradient);
    return -this_gradient + beta * last_direction;
  }

  // Parallel conjugate gradient
  SearchDirection cg_update(const Gradient &last_gradient, const SearchDirection &last_direction,
                            const Gradient &this_gradient, const mpi::communicator& world)
  {
    double beta = dot(this_gradient, this_gradient, world) / dot(last_gradient, last_gradient, world);
    return -this_gradient + beta * last_direction;
  }

  // L-BFGS update search direction
  SearchDirection lbfgs_update(const vector<IterationPoint>& sks, const vector<Gradient>& yks,
                               const Gradient &this_gradient)
  {
    int m = sks.size();

    //cout << "this gradient: " << this_gradient << endl;
    SearchDirection q(this_gradient.begin(), this_gradient.end());
    // temporary coeff in the calculation
    vector<double> rhos(m), alphas(m);
    // left
    for(int i=m-1; i>-1; --i){
      //cout << "yks i: " << yks[i] << endl;
      //cout << "sks i: " << sks[i] << endl;
      //cout << "yks i: " << norm(yks[i]) << "| sks i: " << norm(sks[i]) << endl;
      double rhoi = 1 / dot(yks[i], sks[i]);
      rhos[i] = rhoi;
      double ai = rhoi * dot(sks[i], q);
      alphas[i] = ai;
      //cout << "rho: " << rhoi << " | alpha: " << ai << endl;
      q = q - ai * yks[i];
      //cout << "q norm: " << norm(q) << endl;
    }

    // middle
    // diagonal Hessian0
    SearchDirection hess0(this_gradient.size(), 1.0);
    SearchDirection r = hess0 * q;
    //cout << "r norm: " << norm(r) << endl;

    // right
    for(int i=0; i<m; ++i){
      double beta = rhos[i] * dot(yks[i], r);
      r = r + sks[i] * (alphas[i] - beta);
    }
    //cout << "r norm 2: " << norm(r) << endl;

    return -r;
  }

  // L-BFGS update search direction
  SearchDirection lbfgs_update(const vector<IterationPoint>& sks, const vector<Gradient>& yks,
                               const Gradient &this_gradient, const mpi::communicator& world)
  {
    int m = sks.size();

    //cout << "this gradient: " << this_gradient << endl;
    SearchDirection q(this_gradient.begin(), this_gradient.end());
    // temporary coeff in the calculation
    vector<double> rhos(m), alphas(m);
    // left
    for(int i=m-1; i>-1; --i){
      //cout << "yks i: " << yks[i] << endl;
      //cout << "sks i: " << sks[i] << endl;
      //cout << "yks i: " << norm(yks[i]) << "| sks i: " << norm(sks[i]) << endl;
      double rhoi = 1 / dot(yks[i], sks[i], world);
      rhos[i] = rhoi;
      double ai = rhoi * dot(sks[i], q, world);
      alphas[i] = ai;
      //cout << "rho: " << rhoi << " | alpha: " << ai << endl;
      q = q - ai * yks[i];
      //cout << "q norm: " << norm(q) << endl;
    }

    // middle
    // start with diagonal Hessian0
    SearchDirection hess0(this_gradient.size(), 1.0);
    SearchDirection r = hess0 * q;
    //cout << "r norm: " << norm(r) << endl;

    // right
    for(int i=0; i<m; ++i){
      double beta = rhos[i] * dot(yks[i], r, world);
      r = r + sks[i] * (alphas[i] - beta);
    }
    //cout << "r norm 2: " << norm(r) << endl;

    return -r;
  }

}

// end namespace