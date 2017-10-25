#ifndef OPTIMIZER
#define OPTIMIZER

#include<iostream>
#include<vector>
#include<functional>
#include<cmath>
#include<algorithm>
#include"funcwrapper.h"
#include"utils.h"

class Optimizer{
  public:
    Optimizer(FuncWrapper _fw): func_wrapper(_fw) {};
    ~Optimizer() {};

    //virtual void optimize() = 0;
    virtual double line_search(
        const vector<double> &start_point, const vector<double> &direc,
        vector<double>& end_point, const int max_iter=10000);

  private:
    FuncWrapper func_wrapper;
};

class SteepDescent: public Optimizer {
  public:
    SteepDescent(FuncWrapper _fw): Optimizer(_fw), func_wrapper(_fw) {};
    ~SteepDescent() {};
    virtual vector<vector<double>> optimize(
        const vector<double>& start_point, const double threshold=0.001,
        const int max_iter=10000, const bool verbose=false);
  private:
    FuncWrapper func_wrapper;
};

class ConjugateGradient: public Optimizer {
  public:
    ConjugateGradient(FuncWrapper _fw): Optimizer(_fw), func_wrapper(_fw) {};
    ~ConjugateGradient() {}
    virtual vector<vector<double>> optimize(
        const vector<double>& start_point, const double threshold=0.001,
        const int max_iter=10000, const bool verbose=false);
  private:
    FuncWrapper func_wrapper;
};

class LBGFS: public Optimizer {
  public:
    LBFGS(FuncWrapper _fw): Optimizer(_fw), func_wrapper(_fw) {};
    ~LBFGS() {}
    virtual vector<vector<double>> optimize(
        const vector<double>& start_point, const double threshold=0.001,
        const int max_iter=10000, const bool verbose=false);
  private:
    FuncWrapper func_wrapper;
}

#endif
