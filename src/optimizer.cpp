#include"optimizer.h"

double Optimizer::line_search(
    const vector<double> &start_point,
    const vector<double> &direc,
    vector<double>& end_point,
    const int max_iter)
{
  double v0 = func_wrapper.valueAt(start_point);
  bool find = false;
  int niter = 0;
  double dalpha=0.0001;

  vector<double> incre = direc * dalpha;
  vector<double> p0(start_point), p1(end_point.size());
  while(niter < max_iter){
    //cout << "line search iter: " << niter << endl;
    vector<double> p1 = p0 + incre; 
    //cout << p1 << endl;
    double v1 = func_wrapper.valueAt(p1);
    //cout << "v0 and v1: " << v0 << ", " << v1 << endl;
    if(v1 > v0){
      find = true;
      break;
    }
    std::swap(p0, p1);
    std::swap(v0, v1);
    ++niter;
  }

  end_point = p0;
  double alpha = niter * dalpha;
  return alpha;
}

vector<vector<double>> SteepDescent::optimize(
    const vector<double> start_point,
    const double threshold,
    const int max_iter,
    const bool verbose)
{
  vector<double> x0(start_point.begin(), start_point.end());
  vector<double> x1(start_point.size());
  vector<vector<double>> paths;

  int niter = 0;
  bool converged = false;
  while(!converged && niter < max_iter){
    vector<double> p0 = func_wrapper.gradientAt(x0);
    scale(p0, -1.0);

    if(norm(p0, 2) < threshold){
      converged = true;
    }

    cout << "============> " << niter << endl;
    cout << "x0:" << x0 << "| v0:" << func_wrapper.valueAt(x0) << endl;
    double alpha = line_search(x0, p0, x1);
    double v1 = func_wrapper.valueAt(x1);
    cout << "alpha: " << alpha << endl;
    cout << "x1:" << x1 << "| v1:" << func_wrapper.valueAt(x1) << endl;

    paths.push_back(x1);

    std::swap(x0, x1);
    ++niter;
  }

  return paths;
}

vector<vector<double>> ConjugateGradient::optimize(
    const vector<double> start_point,
    const double threshold,
    const int max_iter,
    const bool verbose)
{
  vector<double> x0(start_point.begin(), start_point.end());
  vector<double> g0 = func_wrapper.gradientAt(x0);
  vector<double> x1(start_point.size());
  vector<double> g1(start_point.size());
  // search direction
  vector<double> p = -g0;
  vector<vector<double>> paths;

  int niter = 0;
  bool converged = false;
  bool restart;
  while(!converged && niter < max_iter){
    paths.push_back(x0);
    cout << "[" << niter << "]x:" << x0 << " | v: "
      << func_wrapper.valueAt(x0) << endl;
    if(norm(g0, 2) < threshold){
      converged = true;
    }

    double alpha = line_search(x0, p, x1);
    g1 = func_wrapper.gradientAt(x1);
    double orth = dot(g0, g1) / dot(g1, g1);
    double beta = dot(g1, g1) / dot(g0, g0);

    if(abs(orth) > 0.1){
      restart = true; 
      p = -g1;
    } else {
      restart = false;
      p = -g1 + p * beta;
    }

    g0 = g1;
    x0 = x1;
    ++niter;
  }

  return paths;
}
