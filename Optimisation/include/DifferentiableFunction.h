#pragma once

#include "LinearOperator.h"
#include <vector>

using std::vector;

/**
 * @brief An example of a differentiable function.
 * There is two functions, one for the value of the function and one for the gradient.
 */
template <typename T>
class DifferentiableFunction
{
public:
    virtual ~DifferentiableFunction() = default;

    virtual double operator()(const vector<T>& x) const = 0;
    virtual vector<T> gradient(const vector<T>& x) const = 0;
};

/**
 * @brief An example of a quadratic function.
 * The quadratic function inherit from my own differentiable function class.
 */
template <typename T>
class Quadratic : public DifferentiableFunction<double>
{
public:
    double operator()(const vector<T>& x) const override { return x[0] * x[0]; }

    vector<T> gradient(const vector<T>& x) const override { return {2 * x[0]}; }
};

/**
 * @brief The Gaussian likelihood function.
 * The GLF function is: f(x,y,phi) = |Phi(x) - y| ^ 2 / (2 * sigma ^ 2)
 * The gradient of the GLF function is: grad(f) = Phi^dagger(Phi(x) - y) / sigma ^ 2
 */
template <typename T>
class GaussianLikelihood : public DifferentiableFunction<double>
{
public:
    GaussianLikelihood(const LinearOperator<double, T>& lineear_op, const vector<T>& y, double sigma);
    ~GaussianLikelihood() = default;

    double operator()(const vector<double>& x) const override;
    vector<double> gradient(const vector<double>& x) const override;

private:
    const LinearOperator<double, T>& phi;
    const vector<T>& y;
    double sigma;
};
