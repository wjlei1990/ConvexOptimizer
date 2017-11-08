//
// Created by Wenjie  Lei on 11/3/17.
//

#include<vector>
#include <boost/mpi.hpp>
#include "vec_utils.hpp"
#include "gtest/gtest.h"
//#include "direcUpdate.h"

namespace mpi = boost::mpi;

using std::vector;
using std::cout;
using std::endl;

TEST(VecUtils, UnaryMinus){
  vector<int> vec{1, 2, 3};
  vector<int> vec3{-1, -2, -3};
  auto vec2 = -vec;
  ASSERT_DOUBLE_EQ(norm(vec2 - vec3), 0.0);
}

TEST(VecUtils, BinaryMinus){
  vector<int> vec1{1, 2, 3};
  vector<int> vec2{1, 2, 3};
  auto vec3 = vec1 - vec2;
  ASSERT_DOUBLE_EQ(norm(vec3), 0.0);
}

TEST(VecUtils, BinaryAdd){
  vector<int> vec1{1, 2, 3};
  vector<int> vec2{3, 2, 1};
  auto vec = vec1 + vec2;
  ASSERT_DOUBLE_EQ(norm(vec - vector<int>{4, 4, 4}), 0.0);
}

TEST(VecUtils, MultiplyScalar){
  vector<double> vec{2, 3, 4};
  int coef = 1;
  auto vec1 = vec * coef;
  auto vec2 = coef * vec;
  ASSERT_DOUBLE_EQ(norm(vec1 - vec), 0);
  ASSERT_DOUBLE_EQ(norm(vec2 - vec), 0);

  double coef1 = 0.5;
  vec1 = vec * coef1;
  vec2 = coef1 * vec;
  ASSERT_DOUBLE_EQ(norm(vec1 - vector<double>{1, 1.5, 2}), 0.0);
  ASSERT_DOUBLE_EQ(norm(vec2 - vector<double>{1, 1.5, 2}), 0.0);
}

TEST(VecUtils, Multiply){
  vector<double> vec1{1, 2, 3};
  vector<double> vec2{3, 2, 1};
  auto vec = vec1 * vec2;

  ASSERT_DOUBLE_EQ(norm(vec - vector<double>{3, 4, 3}), 0.0);
}

TEST(VecUtils, Norm){
  vector<int> vec{-3, 4};

  ASSERT_EQ(norm(vec, 1), 7);
  ASSERT_EQ(norm(vec, 2), 5);
}

TEST(VecUtils, Scale){
  vector<double> vec{0, -1, 2, -3};
  int coef = 2;
  scale(vec, coef);
  ASSERT_DOUBLE_EQ(norm(vec - vector<double>{0, -2, 4, -6}), 0.0);

  double coef2 = 0.5;
  scale(vec, coef2);
  ASSERT_DOUBLE_EQ(norm(vec - vector<double>{0, -1, 2, -3}), 0.0);
}

TEST(VecUtils, Dot){
  vector<int> v1{0, 1, 2, 3};
  vector<int> v2{0, -1, 2, -3};
  auto product = dot(v1, v2);
  ASSERT_DOUBLE_EQ(product, -6);

  vector<double> v3{0, -1, -2, 3};
  auto p2 = dot(v3, v3);
  ASSERT_DOUBLE_EQ(p2, norm(v3) * norm(v3));
}

int main(int argc, char **argv){
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();

  /*
  mpi::environment env;
  mpi::communicator world;

  int rank = world.rank();

  //if(rank == 0) std::cout << "---------> test sd_update" << std::endl;
  //test_sd_update(world);

  //if(rank == 0) std::cout << "---------> test sd_update" << std::endl;
  //test_cg_update(world);

  //test_lbfgs_update(world);
   */
}
