#include"utils.h"

using std::cout;
using std::endl;

std::ostream& operator<<(std::ostream& os, const vector<double>& vec){
  os << "vector [ ";
  for(int i=0; i<vec.size(); ++i){
    os << vec[i] << ", ";
  }
  os << "]";
  return os;
}

vector<double> operator-(const vector<double>& vec){
  vector<double> ret(vec.size());
  for(int i=0; i<vec.size(); ++i){
    ret[i] = -vec[i];
  }
  return ret;
}

vector<double> operator-(const vector<double>& vec1, const vector<double>& vec2){
  vector<double> ret(vec1.size());
  for(int i=0; i<vec1.size(); ++i){
    ret[i] = vec1[i] - vec2[i];
  }
  return ret;
}

vector<double> operator*(const vector<double>& vec, double coef)
{
  vector<double> ret(vec.size());
  for(int i=0; i<vec.size(); ++i){
    ret[i] = coef * vec[i];
  }
  return ret;
}

vector<double> operator*(double coef, const vector<double>& vec){
  return vec * coef;
}

vector<double> operator*(const vector<double>& vec1, const vector<double>& vec2){
  vector<double> ret(vec1.size());
  for(int i=0; i<vec1.size(); ++i){
    ret[i] = vec1[i] * vec2[i];
  }
  return ret;
}

vector<double> operator+(const vector<double>& v1, const vector<double>& v2)
{
  vector<double> v3(v1.size());
  //cout << v1.size() << "," << v2.size() << endl;
  //cout << v1 << endl;
  //cout << v2 << endl;
  for(int i=0; i<v1.size(); ++i){
    v3[i] = v1[i] + v2[i];
    //cout << i << endl;
  }
  return v3;
}

double norm(const vector<double>& vec, const int order){
  double vsum = 0;
  for(auto &v: vec){
    vsum += pow(v, order);
  }
  return pow(vsum, 1.0/order);
}

void scale(vector<double>& vec, double coef){
  for(auto& v: vec){
    v *= coef;
  }
}

double dot(const vector<double>& v1, const vector<double>& v2){
  double sum = 0;
  for(int i=0; i<v1.size(); ++i){
    sum += v1[i] * v2[i];
  }
  return sum;
}
