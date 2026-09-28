#pragma once

#include <complex>
#include <fftw3.h>
#include <memory>
#include <vector>

using std::complex;
using std::shared_ptr;
using std::vector;

/**
 * @brief An abstract class for the linear operator.
 * Here is a forward and backward mapping.
 * @tparam Tx: the type of input image data before apply the operator
 * @tparam Ty: the type of output data after apply the operator
 */
template <typename Tx, typename Ty>
class LinearOperator
{
public:
    virtual ~LinearOperator() = default;

    virtual vector<Ty> operator()(const vector<Tx>& v) const = 0;
    virtual vector<Tx> adjoint(const vector<Ty>& v) const = 0;
};

/**
 * @brief The Identity operator.
 * This operator map the input vector to itself and inherit from my own linear operator base class.
 * @tparam T : the data type of the input data before apply identity operator
 */
template <typename T>
class Identity : public LinearOperator<T, T>
{
public:
    vector<T> operator()(const vector<T>& v) const override { return v; }
    vector<T> adjoint(const vector<T>& v) const override { return v; }
};

/**
 * @brief The sub-sampling operator.
 * This operator receives a vector of data of a given size, and returns a sub-set of that data.
 * Note that the adjoint operator is depending on the size of the input vector.
 * @param indices_input: sub sampling indices input array
 * @param image_size: the origin input image size
 * @tparam T: the type of input image data
 */
template <typename T>
class SubSampling : public LinearOperator<T, T>
{
public:
    SubSampling(const vector<size_t>& indices_input, size_t image_size);
    vector<T> operator()(const vector<T>& x) const override;
    vector<T> adjoint(const vector<T>& y) const override;

private:
    vector<size_t> indices_input;
    size_t x_size;
};

/**
 * @brief The convolution operator.
 * This operator receives a vector of data of a given size, and returns a convolved version of that data.
 * Note that the adjoint operator is depending on the size of the input vector.
 * @param w: the width of the image, using int type here due to the FFTW library
 * @param h: the height of the image, using int type here due to the FFTW library
 * @param eps: the epsilon value for the convolution operator, default as 0.01
 */
class Convolution : public LinearOperator<double, double>
{
public:
    Convolution(const vector<double>& kernel, size_t w, size_t h, double eps = 0.01);
    ~Convolution();

    vector<double> operator()(const vector<double>& x) const override;
    vector<double> adjoint(const vector<double>& y) const override;

private:
    size_t width;
    size_t height;
    size_t N;
    double epsilon;

    // FFTW buffers and plans
    fftw_complex* kernelBuffer;
    fftw_complex* inBuffer;
    fftw_complex* outBuffer;
    fftw_plan forward_plan;
    fftw_plan inverse_plan;
    fftw_plan kernel_plan;

    void computeKernelFFT(const vector<double>& kernel);
};

/**
 * @brief FourierTransform:
 * This operator takes a real image and returns a complex vector
 * via the forward FFT, then the adjoint is the inverse FFT, returning a real vector after rescaling
 * @param width: image width
 * @param height: image height
 */
class FourierTransform : public LinearOperator<double, complex<double>>
{
public:
    FourierTransform(size_t width, size_t height);
    ~FourierTransform();

    vector<complex<double>> operator()(const vector<double>& x) const override;
    vector<double> adjoint(const vector<complex<double>>& y) const override;

private:
    size_t w;
    size_t h;
    size_t N;
    fftw_complex* inBuffer;
    fftw_complex* outBuffer;
    fftw_plan forward_plan;
    fftw_plan inverse_plan;
};

/**
 * @brief Composition operator
 * @param op1: the operator mapping from Tin to Tmid
 * @param op2: the operator mapping from Tmid to Tout
 * @tparam Tin: the type of input data
 * @tparam Tmid: the type of data after apply phi
 * @tparam Tout: the type of output data
 */
template <typename T_in, typename T_mid, typename T_out>
class ComposedOperator : public LinearOperator<T_in, T_out>
{
public:
    ComposedOperator(shared_ptr<LinearOperator<T_in, T_mid>> op1, shared_ptr<LinearOperator<T_mid, T_out>> op2) : phi(op1), psi(op2)
    {
        if (!op1 || !op2)
        {
            throw std::invalid_argument("ERROR: Unexpected null printer input");
        }
    }

    vector<T_out> operator()(const vector<T_in>& x) const override
    {
        return psi->operator()(phi->operator()(x));
    }

    vector<T_in> adjoint(const vector<T_out>& y) const override
    {
        return phi->adjoint(psi->adjoint(y));
    }

private:
    shared_ptr<LinearOperator<T_in, T_mid>> phi;
    shared_ptr<LinearOperator<T_mid, T_out>> psi;
};