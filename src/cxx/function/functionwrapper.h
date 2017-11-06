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

class FunctionWrapper{
  public:
    FunctionWrapper(std::function<double(vector<double>)> _f, int _d): func(_f), dim(_d) {}
    virtual ~FunctionWrapper() {};
    double valueAt(const vector<double>& point);
    vector<double> gradientAt(const vector<double>& point);
    void hess() {};
  private:
    double finite_diff_grad(const vector<double> &point, int idim, double dx);

    std::function<double(vector<double>)> func;
    int dim;
};

#endif
