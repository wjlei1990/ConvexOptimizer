#include<iostream>
#include<vector>
#include"optimizer.h"

using namespace std;

double Rosenbrock(vector<double> arr){
  int a = 1, b = 100;
  double v = std::pow(a - arr[0], 2) + b * std::pow(arr[1] - arr[0] * arr[0], 2);
  return v;
}

int main(){
  FunctionWrapper fw(Rosenbrock, 2);
  vector<double> point = {1, 2};

  cout << fw.valueAt(point) << endl;
  cout << fw.gradientAt(point) << endl;

  //Optimizer::SteepDescent opt(fw);
  //FuncOptimizer::ConjugateGradient opt(fw);
  //FuncOptimizer::LBFGS opt(fw);
  //auto results = opt.optimize(point);
}
