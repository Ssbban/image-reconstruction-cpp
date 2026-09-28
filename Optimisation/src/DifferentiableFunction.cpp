#include "DifferentiableFunction.h"
#include "LinearOperator.h"
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

using std::complex;

// Explicit real number case
template <typename T>
GaussianLikelihood<T>::GaussianLikelihood(const LinearOperator<double, T>& linear_op, const vector<T>& y, double sigma) : phi(linear_op), y(y), sigma(sigma) {}

template <>
double GaussianLikelihood<double>::operator()(const vector<double>& x) const
{
    double magnitude_squared = 0.0;
    auto phi_x = phi(x);

    for (size_t i = 0; i < y.size(); i++)
    {
        double residual = phi_x[i] - y[i];
        magnitude_squared += residual * residual;
    }
    double result = magnitude_squared / (2 * sigma * sigma);
    return result;
}
template <>
vector<double> GaussianLikelihood<double>::gradient(const vector<double>& x) const
{
    vector<double> diff(y.size());
    auto phi_x = phi(x);
    for (size_t i = 0; i < y.size(); i++)
    {
        diff[i] = (phi_x[i] - y[i]) / (sigma * sigma);
    }
    vector<double> result = phi.adjoint(diff);
    return result;
}

// Explicit complex case
template <>
double GaussianLikelihood<complex<double>>::operator()(const vector<double>& x) const
{
    double magnitude_squared = 0.0;
    auto phi_x = phi(x);
    for (size_t i = 0; i < y.size(); i++)
    {
        complex<double> residual = phi_x[i] - y[i];
        magnitude_squared += std::norm(residual);
    }
    return magnitude_squared / (2 * sigma * sigma);
}
template <>
vector<double> GaussianLikelihood<complex<double>>::gradient(const vector<double>& x) const
{
    vector<complex<double>> diff(y.size());
    auto phi_x = phi(x);
    for (size_t i = 0; i < y.size(); i++)
    {
        diff[i] = (phi_x[i] - y[i]) / (sigma * sigma);
    }
    return phi.adjoint(diff);
}
template class GaussianLikelihood<double>;
template class GaussianLikelihood<complex<double>>;