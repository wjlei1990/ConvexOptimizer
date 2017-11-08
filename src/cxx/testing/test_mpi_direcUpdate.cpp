//
// Created by Wenjie  Lei on 11/7/17.
//

#include<vector>
#include <boost/mpi.hpp>
#include "vec_utils.hpp"
#include "gtest/gtest.h"
#include "direcUpdate.h"

using std::cout;
using std::endl;

TEST(ParallelDot, Test) {
  //mpi::environment env;
  mpi::communicator world;
  int rank = world.rank();

  int count = 3;
  std::vector<double> num1(count);
  std::vector<double> num2(count, 1);
  for(int i=0; i<count; ++i)
  {
    num1[i] = count * rank + i;
  }
  int result = dot(num1, num2, world);

  int size = world.size();
  std::vector<double> local_num1(count*size), local_num2(count*size, 1.0);
  for(int i=0; i<local_num1.size(); ++i){
    local_num1[i] = i;
  }
  double local_sum = std::inner_product(local_num1.begin(), local_num1.end(), local_num2.begin(), 0);

  ASSERT_DOUBLE_EQ(local_sum, result);

  world.barrier();
}

TEST(DirectionUpdate, SD_MPI) {
  //mpi::environment env;
  mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  int local_dim = 10;
  int global_dim = local_dim * size;
  // construct the global result on each processor
  vector<double> grad(global_dim);
  for(int i=0; i<global_dim; ++i){
    grad[i] = i;
  }
  vector<double> new_grad = DirectionUpdate::sd_update(grad);

  // construct the local result on each processor
  int start = local_dim * rank;
  int end = local_dim * (rank + 1);
  vector<double> local_grad(grad.begin() + start, grad.begin() + end);
  vector<double> local_new_grad = DirectionUpdate::sd_update(local_grad, world);

  vector<double> true_local_new_grad(new_grad.begin() + start, new_grad.begin() + end);
  ASSERT_DOUBLE_EQ(norm(local_new_grad - true_local_new_grad), 0);

  world.barrier();
}

TEST(DirectionUpdate, CG_MPI)
{
  //mpi::environment env;
  mpi::communicator world;
  int rank = world.rank();
  int size = world.size();
  int local_dim = 10;
  int global_dim = local_dim * size;

  vector<double> grad1(global_dim);
  vector<double> grad2(global_dim);
  vector<double> direc1(global_dim);
  for(int i=0; i<global_dim; ++i) {
    grad1[i] = 2 * i;
    direc1[i] = i;
    grad2[i] = i * i;
  }
  vector<double> new_grad = DirectionUpdate::cg_update(grad1, direc1, grad2);

  int start = local_dim * rank;
  int end = local_dim * (rank + 1);
  vector<double> local_grad1(grad1.begin() + start, grad1.begin() + end);
  vector<double> local_direc1(direc1.begin() + start, direc1.begin() + end);
  vector<double> local_grad2(grad2.begin() + start, grad2.begin() + end);
  vector<double> local_new_grad = DirectionUpdate::cg_update(local_grad1, local_direc1, local_grad2, world);

  vector<double> true_local_new_grad(new_grad.begin() + start, new_grad.begin() + end);
  ASSERT_DOUBLE_EQ(norm(local_new_grad - true_local_new_grad), 0);

  world.barrier();
  //env.finalized();
}

TEST(DirectionUpdate, LBFGS_MPI){
  //mpi::environment env;
  mpi::communicator world;
  int rank = world.rank();
  int size = world.size();
  int local_dim = 5;
  int global_dim = local_dim * size;

  int m = 5;
  vector<vector<double>> sks, local_sks;
  vector<vector<double>> yks, local_yks;
  int start = local_dim * rank;
  int end = local_dim * (rank + 1);
  for(int i=0; i<m; ++i){
    vector<double> g(global_dim), d(global_dim);
    for(int j=0; j<global_dim; ++j){
      g[i] = std::pow(0.5 * (j + m), 0.5);
      d[i] = j + m;
    }
    sks.push_back(g);
    vector<double> local_g(g.begin() + start, g.begin() + end);
    local_sks.push_back(local_g);
    yks.push_back(d);
    vector<double> local_d(d.begin() + start, d.begin() + end);
    local_yks.push_back(local_d);
  }
  vector<double> grad(global_dim);
  for(int i=0; i<global_dim; ++i){
    grad[i] = pow(i * 0.5 + i, 2);
  }
  vector<double> local_grad(grad.begin() + start, grad.begin() + end);

  vector<double> new_grad = DirectionUpdate::lbfgs_update(sks, yks, grad);
  vector<double> true_local_new_grad(new_grad.begin() + start, new_grad.begin() + end);

  vector<double> local_new_grad = DirectionUpdate::lbfgs_update(local_sks, local_yks, local_grad, world);

  ASSERT_DOUBLE_EQ(norm(local_new_grad - true_local_new_grad), 0);

  world.barrier();
}

int main(int argc, char **argv) {
  mpi::environment env;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
