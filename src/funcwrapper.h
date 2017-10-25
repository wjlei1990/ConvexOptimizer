#ifndef FUNCWRAPPER
#define FUNCWRAPPER

#include<cmath>
#include<functional>
#include<iostream>
#include<vector>
#include<stdexcept>

using std::cout;
using std::endl;
using std::vector;

class FuncWrapper{
  public:
    FuncWrapper(std::function<double(vector<double>)> _f, int _d): func(_f), dim(_d) {}
    double valueAt(const vector<double>& point);
    vector<double> gradientAt(const vector<double>& point);
    double hess() {};
  private:
    double finite_diff_grad(const vector<double> &point, int idim, double dx);
    
    std::function<double(vector<double>)> func;
    int dim;
};

#endif
