#pragma once
#include <fftw3.h>

#include <vector>

using std::vector;

/**
 * @brief An abstract class for the non-differentiable function.
 * There is two functions, one for the value of the function and one for the proximal operator.
 * @tparam T: the type of input image data
 */
template <typename T>
class NonDifferentiableFunction
{
public:
    virtual ~NonDifferentiableFunction() = default;

    virtual double operator()(const vector<T>& x) const = 0;
    virtual vector<T> proximal(const vector<T>& x) const = 0;
};

/**
 * @brief An example of Empty non-differentiable function.
 * Inherit from my own non-differentiable function class.
 * @tparam T: the type of input image data
 */
template <typename T>
class Empty : public NonDifferentiableFunction<T>
{
public:
    double operator()(const vector<T>& x) const override
    {
        return 0;
    }

    vector<T> proximal(const vector<T>& x) const override
    {
        return vector<T>(x.size(), 0);
    }
};

/**
 * @brief An example of L1-norm non-differentiable function.
 * @param w: the width of data
 * @param h: the height of data
 * @param tau: the threshold parameter which set default as 0.1
 * @tparam T: the type of input image data
 */
template <typename T>
class L1Norm : public NonDifferentiableFunction<T>
{
public:
    L1Norm(size_t w, size_t h, double tau = 0.1);
    ~L1Norm();

    double operator()(const vector<T>& x) const override;
    vector<T> proximal(const vector<T>& x) const override;

private:
    size_t width;
    size_t height;
    int N;
    double tau;
    double* inBuffer;
    double* outBuffer;
    fftw_plan forward_plan;
    fftw_plan inverse_plan;
};