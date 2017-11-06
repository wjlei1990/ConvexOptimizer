#include<iostream>
#include<vector>
#include"optimizer.h"

using namespace std;

/*
// 2D rosenbrock function
double Rosenbrock(vector<double> arr){
  int a = 1, b = 100;
  double v = std::pow(a - arr[0], 2) + b * std::pow(arr[1] - arr[0] * arr[0], 2);
  return v;
}
 */

// higher order rosenbrock function
double Rosenbrock(vector<double> arr){
  int dim = arr.size();
  int a = 1, b = 100;
  double v = 0.0;
  for(int i=0; i<dim-1; ++i){
    v += std::pow(a - arr[i], 2) + b * std::pow(arr[i+1] - arr[i] * arr[i], 2);
  }
  return v;
}

int main(){
  FunctionWrapper fw(Rosenbrock, 3);
  vector<double> point = {2, 2, 2};

  cout << fw.valueAt(point) << endl;
  cout << fw.gradientAt(point) << endl;

  //FuncOptimizer::SteepDescent opt(fw);
  //FuncOptimizer::ConjugateGradient opt(fw);
  FuncOptimizer::LBFGS opt(fw);
  auto results = opt.optimize(point);
}
