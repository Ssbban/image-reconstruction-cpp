#include "NonDifferentiableFunction.h"
#include <complex>
#include <stdexcept>
#include <vector>

using std::vector;

template <typename T>
L1Norm<T>::L1Norm(size_t w, size_t h, double tau) : N(w * h), width(w), height(h), tau(tau)
{
    // Initialised
    N = w * h;
    inBuffer = (double*)fftw_malloc(sizeof(double) * N);
    outBuffer = (double*)fftw_malloc(sizeof(double) * N);

    if (!inBuffer || !outBuffer)
    {
        throw std::runtime_error("ERROR: fftw_malloc the initial buffer failed");
    }

    forward_plan = fftw_plan_r2r_2d(
        h,
        w,
        inBuffer,
        outBuffer,
        FFTW_REDFT10,
        FFTW_REDFT10,
        FFTW_MEASURE);

    inverse_plan = fftw_plan_r2r_2d(
        h,
        w,
        inBuffer,
        outBuffer,
        FFTW_REDFT01,
        FFTW_REDFT01,
        FFTW_MEASURE);
}

template <typename T>
L1Norm<T>::~L1Norm()
{
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
    fftw_free(inBuffer);
    fftw_free(outBuffer);
}

template <typename T>
double L1Norm<T>::operator()(const vector<T>& x) const
{
    for (int i = 0; i < N; i++)
    {
        inBuffer[i] = x[i];
    }

    fftw_execute(forward_plan);

    double sum_abs = 0.0;
    for (int i = 0; i < N; i++)
    {
        sum_abs += std::abs(outBuffer[i]);
    }
    return sum_abs;
}

template <typename T>
vector<T> L1Norm<T>::proximal(const vector<T>& x) const
{
    for (int i = 0; i < N; i++)
    {
        inBuffer[i] = x[i];
    }

    fftw_execute(forward_plan);

    // Determine S(~x_i)
    for (int i = 0; i < N; i++)
    {
        double value = outBuffer[i];
        double abs_value = std::abs(value);
        if (abs_value <= tau)
        {
            inBuffer[i] = 0.0;
        }
        else
        {
            inBuffer[i] = value * (1.0 - (tau / abs_value));
        }
    }

    fftw_execute(inverse_plan);

    // Normalised and return diff
    vector<double> modify_imag(N);
    vector<double> diff_result(N);
    for (int i = 0; i < N; i++)
    {
        modify_imag[i] = outBuffer[i] / (4 * N);
        diff_result[i] = x[i] - modify_imag[i];
    }
    return diff_result;
}
template class L1Norm<double>;