#ifndef UTILS
#define UTILS

#include<iostream>
#include<vector>
#include<cmath>

using std::vector;

std::ostream& operator<<(std::ostream& os, const vector<double>& vec);

vector<double> operator-(const vector<double>& vec);
vector<double> operator-(const vector<double>& vec1, const vector<double>& vec2);

vector<double> operator*(const vector<double>& vec1, const vector<double>& vec2);
vector<double> operator*(const vector<double>& vec, double coef);
vector<double> operator*(double coef, const vector<double>& vec);

vector<double> operator+(const vector<double>& v1, const vector<double>& v2);
double norm(const vector<double>& vec, const int order=2);

void scale(vector<double>& vec, double coef);

double dot(const vector<double>&, const vector<double>&);

#endif
