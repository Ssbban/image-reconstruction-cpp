#include "LinearOperator.h"
#include <stdexcept>
#include <vector>

//--------------------------------------------- The Sub-sampling operator implementation ---------------------------------------
template <typename T>
SubSampling<T>::SubSampling(const vector<size_t>& indices, size_t image_size) : indices_input(indices), x_size(image_size)
{
    // Error handeling
    if (indices_input.empty())
    {
        throw std::invalid_argument("ERROR: Unexpected empty indices");
    }
    for (auto idx : indices_input)
    {
        if (idx >= image_size)
        {
            throw std::out_of_range("ERROR: Index size exceeds image size");
        }
    }
}

template <typename T>
vector<T> SubSampling<T>::operator()(const vector<T>& x) const
{
    vector<T> y(indices_input.size());
    for (size_t i = 0; i < indices_input.size(); i++)
    {
        y[i] = x[indices_input[i]];
    }
    return y;
}

template <typename T>
vector<T> SubSampling<T>::adjoint(const vector<T>& y) const
{
    vector<T> x_prime(x_size, T(0));
    for (size_t i = 0; i < indices_input.size(); i++)
    {
        x_prime[indices_input[i]] = y[i];
    }
    return x_prime;
}

template class SubSampling<double>;
template class SubSampling<complex<double>>;

//--------------------------------------------- Convolution Operator Implementation ------------------------------------------------------
Convolution::Convolution(const vector<double>& kernel, size_t w, size_t h, double eps)
    : width(w),
      height(h),
      N(w * h),
      epsilon(eps)
{
    if (kernel.size() != N)
    {
        throw std::invalid_argument("ERROR: Convolution: kernel size must be the same as image size");
    }

    // allocate buffers memory and initialised plan
    kernelBuffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    inBuffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    outBuffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    if (!kernelBuffer || !inBuffer || !outBuffer)
    {
        throw std::runtime_error("ERROR: fftw_malloc failed");
    }

    forward_plan = fftw_plan_dft_2d(
        height,
        width,
        inBuffer,
        outBuffer,
        FFTW_FORWARD,
        FFTW_MEASURE);
    if (!forward_plan)
    {
        throw std::runtime_error("ERROR: Failed to create forwardplan");
    }

    inverse_plan = fftw_plan_dft_2d(
        height,
        width,
        inBuffer,
        outBuffer,
        FFTW_BACKWARD,
        FFTW_MEASURE);
    if (!inverse_plan)
    {
        throw std::runtime_error("ERROR: Failed to create inverseplan");
    }

    // determine the forward fft for kernel
    computeKernelFFT(kernel);
    for (size_t i = 0; i < N; i++)
    {
        kernelBuffer[i][0] = outBuffer[i][0];
        kernelBuffer[i][1] = outBuffer[i][1];
    }
}

Convolution::~Convolution()
{
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
    fftw_free(kernelBuffer);
    fftw_free(inBuffer);
    fftw_free(outBuffer);
}

void Convolution::computeKernelFFT(const vector<double>& kernel)
{
    for (size_t i = 0; i < N; i++)
    {
        inBuffer[i][0] = kernel[i];
        inBuffer[i][1] = 0.0;
    }

    fftw_execute(forward_plan);

    for (size_t i = 0; i < N; i++)
    {
        kernelBuffer[i][0] = outBuffer[i][0];
        kernelBuffer[i][1] = outBuffer[i][1];
    }
}

vector<double> Convolution::operator()(const vector<double>& x) const
{
    if (x.size() != N)
    {
        throw std::invalid_argument("ERROR: Input x size must be the same as image size");
    }

    // Implement the forward fft for input x
    for (size_t i = 0; i < N; i++)
    {
        inBuffer[i][0] = x[i];
        inBuffer[i][1] = 0.0;
    }
    fftw_execute(forward_plan);

    // The complex product for x~ and k~
    for (size_t i = 0; i < N; i++)
    {
        inBuffer[i][0] = (outBuffer[i][0] * kernelBuffer[i][0]) - (outBuffer[i][1] * kernelBuffer[i][1]);
        inBuffer[i][1] = (outBuffer[i][0] * kernelBuffer[i][1]) + (outBuffer[i][1] * kernelBuffer[i][0]);
    }

    fftw_execute(inverse_plan);

    // Normalise
    vector<double> result(N);
    for (size_t i = 0; i < N; i++)
    {
        result[i] = outBuffer[i][0] / N;
    }
    return result;
}

vector<double> Convolution::adjoint(const vector<double>& y) const
{
    if (y.size() != N)
    {
        throw std::invalid_argument("ERROR: The size of adjoint input y must be the same as image size");
    }

    // Implement the forward fft for input y
    for (size_t i = 0; i < N; i++)
    {
        inBuffer[i][0] = y[i];
        inBuffer[i][1] = 0.0;
    }
    fftw_execute(forward_plan);

    for (size_t i = 0; i < N; i++)
    {
        double xr = outBuffer[i][0];
        double xi = outBuffer[i][1];
        double kr = kernelBuffer[i][0];
        double ki = kernelBuffer[i][1];

        // Calculate ~x_i = ~x'_i / ~k_i
        inBuffer[i][0] = (xr * kr + xi * ki) / (kr * kr + ki * ki + epsilon);
        inBuffer[i][1] = (xi * kr - xr * ki) / (kr * kr + ki * ki + epsilon);
    }

    fftw_execute(inverse_plan);

    // Normalise
    vector<double> result(N);
    for (size_t i = 0; i < N; i++)
    {
        result[i] = outBuffer[i][0] / N;
    }
    return result;
}

// -------------------------------------------- Fourier transform Implementation ------------------------------------------------------
FourierTransform::FourierTransform(size_t width, size_t height) : w(width), h(height), N(width * height)
{
    if (N == 0)
    {
        throw std::invalid_argument("ERROR: Unexpected zero dimension given is not allowed for FourierTransform");
    }

    // allocated the memery and initialised plan
    inBuffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    outBuffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
    if (!inBuffer || !outBuffer)
    {
        throw std::runtime_error("ERROR: fftw_malloc failed");
    }

    forward_plan = fftw_plan_dft_2d(
        h,
        w,
        inBuffer,
        outBuffer,
        FFTW_FORWARD,
        FFTW_MEASURE);
    if (!forward_plan)
    {
        throw std::runtime_error("ERROR: Failed to create forward plan");
    }

    inverse_plan = fftw_plan_dft_2d(
        h,
        w,
        inBuffer,
        outBuffer,
        FFTW_BACKWARD,
        FFTW_MEASURE);
    if (!inverse_plan)
    {
        throw std::runtime_error("ERROR: Failed to create inverse plan");
    }
}

FourierTransform::~FourierTransform()
{
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
    fftw_free(inBuffer);
    fftw_free(outBuffer);
}

vector<complex<double>> FourierTransform::operator()(const vector<double>& x) const
{
    if (x.size() != N)
    {
        throw std::runtime_error("ERROR: The input size of FourierTransform mismatch");
    }

    // Implement the forward fft for input x
    for (size_t i = 0; i < N; i++)
    {
        inBuffer[i][0] = x[i];
        inBuffer[i][1] = 0.0;
    }
    fftw_execute(forward_plan);

    vector<complex<double>> result(N);
    for (size_t i = 0; i < N; i++)
    {
        result[i] = complex<double>(outBuffer[i][0], outBuffer[i][1]);
    }
    return result;
}

vector<double> FourierTransform::adjoint(const vector<complex<double>>& y) const
{
    if (y.size() != N)
    {
        throw std::runtime_error("ERROR: FourierTransform adjoint input size mismatch");
    }

    // Implement the inverse fft and normalised
    for (size_t i = 0; i < N; i++)
    {
        inBuffer[i][0] = y[i].real();
        inBuffer[i][1] = y[i].imag();
    }
    fftw_execute(inverse_plan);

    // Normalised
    vector<double> result(N);
    for (size_t i = 0; i < N; i++)
    {
        result[i] = outBuffer[i][0] / N;
    }

    return result;
}