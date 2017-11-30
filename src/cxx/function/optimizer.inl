#include"optimizer.h"
#define FLOAT_TYPE double

namespace FuncOptimizer {

  template<typename T>
  T FunctionOptimizer<T>::line_search(
      const IterationPoint<T>& start_point, const SearchDirection<T>& direc,
      IterationPoint<T>& end_point, const int max_iter) {
    double v0 = func_wrapper.valueAt(start_point);
    bool find = false;
    int niter = 0;
    double dalpha = 0.0001;

    SearchDirection<T> incre = direc * dalpha;
    IterationPoint<T> p0(start_point), p1(end_point.size());
    while (niter < max_iter) {
      //cout << "line search iter: " << niter << endl;
      SearchDirection<T> p1 = p0 + incre;
      //cout << p1 << endl;
      double v1 = func_wrapper.valueAt(p1);
      //cout << "v0 and v1: " << v0 << ", " << v1 << endl;
      if (v1 > v0) {
        find = true;
        break;
      }
      std::swap(p0, p1);
      std::swap(v0, v1);
      ++niter;
    }
    //cout << "niter in line search: " << niter << endl;

    end_point = p0;
    double alpha = niter * dalpha;
    return alpha;
  }

  template<typename T>
  OptimResult<T> SteepDescent<T>::optimize(
      const IterationPoint<T> start_point, const double threshold,
      const int max_iter, const bool verbose) {
    IterationPoint<T> x0(start_point.begin(), start_point.end());
    SearchDirection<T> g0;
    IterationPoint<T> x1(start_point.size());
    vector<IterationPoint<T>> paths;

    OptimResult<T> result;
    int niter = 0;
    bool converged = false;
    while (!converged && niter < max_iter) {
      paths.push_back(x0);
      g0 = func_wrapper.gradientAt(x0);
      if (norm(g0, 2) < threshold) {
        converged = true;
        break;
      }
      SearchDirection<T> p0 = DirectionUpdate::sd_update(g0);

      //cout << "============> " << niter << endl;
      //cout << "x0:" << x0 << "| v0:" << func_wrapper.valueAt(x0) << endl;
      double alpha = line_search(x0, p0, x1);
      double v1 = func_wrapper.valueAt(x1);
      //cout << "alpha: " << alpha << endl;
      //cout << "x1:" << x1 << "| v1:" << func_wrapper.valueAt(x1) << endl;

      std::swap(x0, x1);
      ++niter;
    }

    result.converged = converged;
    result.path = paths;
    return result;
  }

  template<typename T>
  OptimResult<T> ConjugateGradient<T>::optimize(
      const IterationPoint<T> start_point, const double threshold,
      const int max_iter, const bool verbose) {
    IterationPoint<T> x0(start_point.begin(), start_point.end());
    SearchDirection<T> g0 = func_wrapper.gradientAt(x0);
    IterationPoint<T> x1(start_point.size());
    SearchDirection<T> g1(start_point.size());
    // search direction
    SearchDirection<T> p = -g0;
    vector<IterationPoint<T>> paths;

    OptimResult<T> result;
    int niter = 0;
    bool converged = false;
    bool restart;
    while (!converged && niter < max_iter) {
      paths.push_back(x0);
      if (norm(g0, 2) < threshold) {
        converged = true;
        break;
      }

      double alpha = line_search(x0, p, x1);
      g1 = func_wrapper.gradientAt(x1);
      double orth = dot(g0, g1) / dot(g1, g1);

      if (abs(orth) > 0.1) {
        restart = true;
        p = DirectionUpdate::sd_update(g1);
      } else {
        restart = false;
        p = DirectionUpdate::cg_update(g0, p, g1);
      }

      //cout << "[" << niter << "] x: " << x0 << " | v: " << func_wrapper.valueAt(x0)
      //     << " | norm: " << norm(g1) << " | orth: " << orth << " | restart: " << restart << endl;
      g0 = g1;
      x0 = x1;
      ++niter;
    }

    result.converged = converged;
    result.path = paths;
    return result;
  }

  template<typename T>
  OptimResult<T> LBFGS<T>::optimize(
      const IterationPoint<T> start_point, const double threshold,
      const int max_iter, const bool verbose, int memory)
  {
    IterationPoint<T> x0 = start_point;
    Gradient<T> g0 = func_wrapper.gradientAt(x0);
    IterationPoint<T> x1(start_point.size());
    Gradient<T> g1(start_point.size());
    vector<IterationPoint<T>> paths;

    OptimResult<T> result;
    int niter = 0;
    bool converged = false;
    SearchDirection<T> p;
    vector<IterationPoint<T>> sks;
    vector<Gradient<T>> yks;
    while(!converged && niter < max_iter){
      //cout << "========================>\n [" << niter << "]" << endl;
      paths.push_back(x0);
      if(norm(g0, 2) < threshold){
        converged = true;
        break;
      }

      if(niter == 0){
        p = DirectionUpdate::sd_update(g0);
      } else {
        p = DirectionUpdate::lbfgs_update(sks, yks, g0);
      }

      double alpha = line_search(x0, p, x1);

      //cout << "---> summary:" << endl;
      //cout << "x0: " << x0 << " | v0: " << func_wrapper.valueAt(x0) << endl;
      //cout << "x1: " << x1 << " | v1: " << func_wrapper.valueAt(x1) << endl;
      //cout << "diff norm" << norm(x1 - x0) << endl;
      //cout << "norm:" << norm(g0) << endl;

      // clean the sks and yks which are m iterations before
      if(niter >= memory){
        yks.erase(yks.begin());
        sks.erase(sks.begin());
      }
      sks.push_back(x1 - x0);
      g1 = func_wrapper.gradientAt(x1);
      yks.push_back(g1 - g0);

      x0 = x1;
      g0 = g1;
      ++niter;
    }

    result.converged = converged;
    result.path = paths;
    return result;
  }
} // end namespace
