//
// Created by Wenjie  Lei on 11/7/17.
//
#include <vector>
#include <cmath>
#include "gtest/gtest.h"
#include "optimizer.h"

using std::vector;

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

TEST(RosenBrock, Test) {
  ASSERT_DOUBLE_EQ(Rosenbrock(vector<double>{1, 1}), 0);
  ASSERT_DOUBLE_EQ(Rosenbrock(vector<double>{1, 2}), 100);

  ASSERT_DOUBLE_EQ(Rosenbrock(vector<double>{1, 1, 1}), 0);
  ASSERT_DOUBLE_EQ(Rosenbrock(vector<double>{1, 1, 2}), 100);

  ASSERT_DOUBLE_EQ(Rosenbrock(vector<double>{1, 1, 1, 1, 1, 1}), 0);
}

TEST(Optimize, SteepDescent){
  FunctionWrapper fw(Rosenbrock, 3);
  vector<double> point = {2, 2, 2};

  FuncOptimizer::SteepDescent opt(fw);
  auto results = opt.optimize(point);
}

TEST(Optimize, ConjugateGradient){
  FunctionWrapper fw(Rosenbrock, 3);
  vector<double> point = {2, 2, 2};

  FuncOptimizer::SteepDescent opt(fw);
  auto results = opt.optimize(point);
}

TEST(Optimize, LBFGS){
  FunctionWrapper fw(Rosenbrock, 3);
  vector<double> point = {2, 2, 2};

  FuncOptimizer::LBFGS opt(fw);
  auto results = opt.optimize(point);
}
