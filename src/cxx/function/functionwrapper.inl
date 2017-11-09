template<typename T>
T FunctionWrapper<T>::valueAt(const vector<T>& point){
  if(point.size() != dim){
    throw std::invalid_argument("Dimension of input arr is wrong!");
  }
  return func(point);
}

template<typename T>
T FunctionWrapper<T>::finite_diff_grad(const vector<T> &point, int idim, T dx){
  vector<T> x0(point), x1(point);
  x0[idim] -= dx;
  x1[idim] += dx;

  T grad = (valueAt(x1) - valueAt(x0)) / (2 * dx);

  return grad;
}

template<typename T>
vector<T> FunctionWrapper<T>::gradientAt(const vector<T> &point){
  T dx = 0.00001;
  vector<T> grad(point.size(), 0.0);
  
  for(int i=0; i<point.size(); ++i){
    grad[i] = finite_diff_grad(point, i, dx);
  }
  return grad;
}
