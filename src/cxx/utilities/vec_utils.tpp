#include "vec_utils.hpp"

template <class T>
std::ostream& operator<<(std::ostream& os, const vector<T>& v){
  if(!v.empty()) {
    os << "Vector[ ";
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(os, ", "));
    os << "\b\b]";
  }
  return os;
}

template <class T>
vector<T> operator-(const vector<T>& vec){
  vector<T> ret(vec.size());
  for(int i=0; i<vec.size(); ++i){
    ret[i] = -vec[i];
  }
  return ret;
}

template <class T>
vector<T> operator-(const vector<T>& vec1, const vector<T>& vec2){
  vector<T> ret(vec1.size());
  for(int i=0; i<vec1.size(); ++i){
    ret[i] = vec1[i] - vec2[i];
  }
  return ret;
}

template <class T1, class T2>
vector<T1> operator*(const vector<T1>& vec, T2 coef)
{
  vector<T1> ret(vec.size());
  for(int i=0; i<vec.size(); ++i){
    ret[i] = coef * vec[i];
  }
  return ret;
}

template <class T1, class T2>
vector<T2> operator*(T1 coef, const vector<T2>& vec){
  return vec * coef;
}

template <class T>
vector<T> operator*(const vector<T>& vec1, const vector<T>& vec2){
  vector<T> ret(vec1.size());
  for(int i=0; i<vec1.size(); ++i){
    ret[i] = vec1[i] * vec2[i];
  }
  return ret;
}

template <class T>
vector<T> operator+(const vector<T>& v1, const vector<T>& v2)
{
  vector<T> v3(v1.size());
  for(int i=0; i<v1.size(); ++i){
    v3[i] = v1[i] + v2[i];
  }
  return v3;
}

template <class T>
T norm(const vector<T>& vec, const int order){
  T vsum = 0;
  for(auto &v: vec){
    vsum += pow(v, order);
  }
  return pow(vsum, 1.0/order);
}

template <class T>
void scale(vector<T>& vec, T coef){
  for(auto& v: vec){
    v *= coef;
  }
}

template <class T>
T dot(const vector<T>& v1, const vector<T>& v2){
  return std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0);
}

template <class T>
T dot(const vector<T>& v1, const vector<T>& v2, const mpi::communicator& world) {
  T local_sum = std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0);
  T global_sum;
  mpi::all_reduce(world, local_sum, global_sum, std::plus<T>());
  return global_sum;
}
