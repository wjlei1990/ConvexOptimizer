#include"functionwrapper.h"

double FunctionWrapper::valueAt(const vector<double>& point){
  if(point.size() != dim){
    throw std::invalid_argument("Dimension of input arr is wrong!");
  }
  return func(point);
}

double FunctionWrapper::finite_diff_grad(const vector<double> &point, int idim, double dx){
  vector<double> x0(point), x1(point);
  x0[idim] -= dx;
  x1[idim] += dx;

  double grad = (valueAt(x1) - valueAt(x0)) / (2 * dx);

  return grad;
}

vector<double> FunctionWrapper::gradientAt(const vector<double> &point){
  double dx = 0.00001;
  vector<double> grad(point.size(), 0.0);
  
  for(int i=0; i<point.size(); ++i){
    grad[i] = finite_diff_grad(point, i, dx);
  }
  return grad;
}
