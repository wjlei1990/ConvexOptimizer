#ifndef UTILS
#define UTILS

#include <iostream>
#include <vector>
#include <cmath>
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <numeric>


using std::vector;
namespace mpi = boost::mpi;

template<class T>
std::ostream& operator<<(std::ostream& os, const vector<T>& vec);

template<class T>
vector<T> operator-(const vector<T>& vec);

template<class T>
vector<T> operator-(const vector<T>& vec1, const vector<T>& vec2);

template<class T>
vector<T> operator+(const vector<T>& v1, const vector<T>& v2);

template<class T>
vector<T> operator*(const vector<T>& vec1, const vector<T>& vec2);

template<class T1, class T2>
vector<T1> operator*(const vector<T1>& vec, T2 coef);

template<class T1, class T2>
vector<T2> operator*(T1 coef, const vector<T2>& vec);

template<class T>
double norm(const vector<T>& vec, const int order=2);

template<typename T1, typename T2>
void scale(vector<T1>& vec, T2 coef);

template<class T>
T dot(const vector<T>&, const vector<T>&);

template<class T>
T dot(const vector<T>& v1, const vector<T>& v2, const mpi::communicator& world);

#include "vec_utils.inl"
#endif


//


