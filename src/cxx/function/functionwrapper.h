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

template<typename T>
class FunctionWrapper{
  public:
    FunctionWrapper(std::function<T(vector<T>)> _f, int _d): func(_f), dim(_d) {}
    virtual ~FunctionWrapper() {};
    T valueAt(const vector<T>& point);
    vector<T> gradientAt(const vector<T>& point);
    void hess() {};
  private:
    T finite_diff_grad(const vector<T> &point, int idim, T dx);

    std::function<T(vector<T>)> func;
    int dim;
};

#include "functionwrapper.inl"

#endif
