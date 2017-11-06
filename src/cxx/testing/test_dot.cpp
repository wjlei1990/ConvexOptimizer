//
// Created by Wenjie  Lei on 11/3/17.
//

#include<vector>
#include <boost/mpi.hpp>
#include "vec_utils.hpp"
//#include "direcUpdate.h"

namespace mpi = boost::mpi;

using std::vector;
using std::cout;
using std::endl;

void test_unary_minus(){
  vector<int> vec{1, 2, 3};
  auto vec2 = -vec;
  cout << vec2 << endl;
}

void test_binary_minus(){
  vector<int> vec1{1, 2, 3};
  vector<int> vec2{1, 2, 3};
  auto vec3 = vec1 - vec2;
  cout << vec3 << endl;
}

void test_multiply_1(){
  vector<int> vec{1, 2, 3};
  int coef = 2;

  auto vec1 = vec * coef;
  auto vec2 = coef * vec;

  cout << vec1 << endl;
  cout << vec2 << endl;

}


int test_dot(const mpi::communicator& world)
{
  int rank = world.rank();

  int count = 3;
  std::vector<double> num1(count, 0);
  std::vector<double> num2(count, 1);
  for(int i=0; i<count; ++i)
  {
    num1[i] = count * rank + i;
  }

  int result = dot<double>(num1, num2, world);

  int size = world.size();
  std::vector<double> local_num1(count*size), local_num2(count*size, 1.0);
  for(int i=0; i<local_num1.size(); ++i){
    local_num1[i] = i;
  }
  double local_sum = std::inner_product(local_num1.begin(), local_num1.end(), local_num2.begin(), 0);

  std::cout << "rank: " << rank << " | " << result << "| check: " << local_sum << std::endl;

  return 1;
}

/*
void test_sd_update(const mpi::communicator& world)
{
  int rank = world.rank();
  int size = world.size();
  int local_dim = 2;
  int global_dim = local_dim * size;

  vector<double> grad(global_dim);
  for(int i=0; i<global_dim; ++i) {
    grad[i] = i;
  }
  vector<double> new_grad = DirectionUpdate::sd_update(grad);

  int start = local_dim * rank;
  int end = local_dim * (rank + 1);
  vector<double> true_local_new_grad(new_grad.begin() + start, new_grad.begin() + end);

  vector<double> local_grad(grad.begin() + start, grad.begin() + end);
  vector<double> local_new_grad = DirectionUpdate::sd_update(local_grad, world);

  std::cout << "rank: " << rank << " | " << true_local_new_grad << " | " << local_new_grad << endl;
}

void test_cg_update(const mpi::communicator& world)
{
  int local_dim = 2;
  int rank = world.rank();
  int size = world.size();
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
  vector<double> true_local_new_grad(new_grad.begin() + start, new_grad.begin() + end);

  vector<double> local_grad1(grad1.begin() + start, grad1.begin() + end);
  vector<double> local_direc1(direc1.begin() + start, direc1.begin() + end);
  vector<double> local_grad2(grad2.begin() + start, grad2.begin() + end);
  vector<double> local_new_grad = DirectionUpdate::cg_update(local_grad1, local_direc1, local_grad2, world);

  std::cout << "rank: " << rank << " | " << true_local_new_grad << " | " << local_new_grad << endl;
}

void test_lbfgs_update(const mpi::communicator& world){
  int local_dim = 5;
  int rank = world.rank();
  int size = world.size();
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
  //assign local variables
  std::cout << "rank: " << rank << " | " << true_local_new_grad << " | " << local_new_grad << endl;
}
*/


int main(){
  mpi::environment env;
  mpi::communicator world;

  int rank = world.rank();

  test_unary_minus();
  test_binary_minus();
  test_multiply_1();

  if(rank == 0) std::cout << "---------> test dot" << std::endl;
  test_dot(world);

  //if(rank == 0) std::cout << "---------> test sd_update" << std::endl;
  //test_sd_update(world);

  //if(rank == 0) std::cout << "---------> test sd_update" << std::endl;
  //test_cg_update(world);

  //test_lbfgs_update(world);
}
