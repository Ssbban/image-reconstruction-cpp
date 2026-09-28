#include "DataPack.h"
#include "DifferentiableFunction.h"
#include "ImageUtils.h"
#include "LinearOperator.h"
#include "NonDifferentiableFunction.h"
#include "OptimisationUtils.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <complex>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>

using namespace Catch::Matchers;
using std::invalid_argument;
using std::out_of_range;
using std::runtime_error;

/****************************************************************************/
/* Section 1.2 - IO Testing
/****************************************************************************/

/**
 * @brief Test the possible error on ReadData function
 */
// TODO: Test your IO functions here
TEST_CASE("File not opened", "I/O Test")
{
    REQUIRE_THROWS_AS(OptimisationUtils::ReadData<double>("test/data/nothisfile.dat"), runtime_error);
}

TEST_CASE("Empty file", "I/O Test")
{
    REQUIRE_THROWS_AS(OptimisationUtils::ReadData<double>("test/test_file/TestSphere_proximal/empty_file.dat"), runtime_error);
}

TEST_CASE("Dimension = 2", "I/O Test")
{
    REQUIRE_THROWS_AS(OptimisationUtils::ReadData<double>("test/test_file/TestSphere_proximal/3d_first_line.dat"), invalid_argument);
}

TEST_CASE("First line is valid input", "I/O Test")
{
    REQUIRE_THROWS_AS(OptimisationUtils::ReadData<double>("test/test_file/TestSphere_proximal/first_line_invalid.dat"), invalid_argument);
}

TEST_CASE("First line out of range", "I/O Test")
{
    REQUIRE_THROWS_AS(OptimisationUtils::ReadData<double>("test/test_file/TestSphere_proximal/out_of_range.dat"), out_of_range);
}

TEST_CASE("Second line exists", "I/O Test")
{
    REQUIRE_THROWS_AS(OptimisationUtils::ReadData<double>("test/test_file/TestSphere_proximal/second_line_empty.dat"), runtime_error);
}

TEST_CASE("Second line format", "I/O Test")
{
    REQUIRE_THROWS_AS(OptimisationUtils::ReadData<double>("test/test_file/TestSphere_proximal/second_line_invalid.dat"), invalid_argument);
    REQUIRE_THROWS_AS(OptimisationUtils::ReadData<double>("test/test_file/TestSphere_proximal/two_arg_second_line.dat"), invalid_argument);
}

TEST_CASE("Second line out of range", "I/O Test")
{
    REQUIRE_THROWS_AS(OptimisationUtils::ReadData<double>("test/test_file/TestSphere_proximal/second_line_out_of_range.dat"), out_of_range);
}

TEST_CASE("Test the overall function is valided", "I/O Test")
{
    DataPack<double> data_double = OptimisationUtils::ReadData<double>("test/data/measurements/TestSphere_proximal.dat");
    DataPack<complex<double>> data_complex = OptimisationUtils::ReadData<complex<double>>("test/test_file/TestSphere_proximal/complex_data.dat");
}

/****************************************************************************/
/* Section 2.3 - Optimisation Test
/****************************************************************************/

/**
 * @brief Test the optimisation algorithm for a simple quadratic function
 */
TEST_CASE("Test optimisation algorithm for simple quadratic function", "Algorithm Tests")
{
    // TODO: Make sure Quadratic is a differentiable function type
    Quadratic<double> f_quadratic;

    // TODO: Make sure Empty is a non-differentiable function type
    Empty<double> g_empty;

    // TODO: Make sure Identity is a linear operator type
    Identity<double> Id;

    // Quadratic works with vectors of size 1 to represent a single value
    vector<double> y = {1.0}; // because Phi = Id the value of y is the start value for x

    // TODO: call your optimisation function with the default convergence function
    double alpha = 0.1;
    double beta = 0.1;
    size_t n_max = 1000;
    double delta = 1e-6;
    vector<double> result = OptimisationUtils::IterativeAlgorithm(f_quadratic, g_empty, y, Id, alpha, beta, n_max, delta);

    // Check that result is close to 0
    REQUIRE(result.size() == 1);
    REQUIRE(std::abs(result[0]) < 1e-3);
}

/****************************************************************************/
/* Section 3.1 - Linear Operator Testing
/****************************************************************************/

// TODO: Write a test to check your sub-sampler operator
TEST_CASE("Test sub-sampler", "Linear Operator Tests")
{
    // Constructe subsampling
    vector<size_t> indices_I = {1, 4, 2};
    vector<double> input = {0, 9, 4, 1, 2, 5};
    SubSampling<double> sampler(indices_I, input.size());

    // Check the operator function
    vector<double> expected_y = {9, 2, 4};
    REQUIRE(sampler(input) == expected_y);

    // Check the adjoint function
    vector<double> expected_xprime = {0, 9, 4, 0, 2, 0};
    vector<double> adjoint_result = sampler.adjoint(sampler(input));
    REQUIRE(adjoint_result == expected_xprime);

    // Test the unexpected input
    vector<size_t> empty_indices;
    REQUIRE_THROWS_AS(SubSampling<double>(empty_indices, 5), invalid_argument);

    vector<size_t> indices_out_of_range = {1, 5, 10};
    REQUIRE_THROWS_AS(SubSampling<double>(indices_out_of_range, 5), out_of_range);

    REQUIRE_THROWS_AS(OptimisationUtils::ReadData<double>("test/test_file/TestSphere_proximal/second_line_out_of_range.dat"), out_of_range);
}

// TODO: Complete this convolution test function using your class
TEST_CASE("Test Convolution", "Linear Operator Tests")
{
    // TODO: Load the following image as your start image
    string image_path = "test/data/images/TestSphere.pgm";
    DataPack true_image = ImageUtils::ReadImage<double>(image_path);

    vector<double> kernel = ImageUtils::GenSincKernel(true_image.width, true_image.height, 1.0);

    // TODO: Define and apply your convolution and check that the output matches the following image file
    // Determine the blurred image
    Convolution ConvolutionOp(kernel, true_image.width, true_image.height);
    vector<double> blurred = ConvolutionOp(true_image.data);
    DataPack<double> blurred_datapack{true_image.width, true_image.height, blurred};
    ImageUtils::WriteImage<double>(blurred_datapack, "test/test_file/TestSphere_convolution/conv.pgm");

    // Check the blurred image size
    string blurred_image_path = "test/data/images/TestSphere_convolved.pgm";
    DataPack blurred_image_expected = ImageUtils::ReadImage<double>(blurred_image_path);
    REQUIRE(blurred_datapack.data.size() == blurred_image_expected.data.size());

    // TODO: Check that your blurred image is close to the expecation
    // Possible that there are some system dependent issues, thus larger torlerance
    int tol = 35;
    for (size_t i = 0; i < blurred.size(); i++)
    {
        REQUIRE_THAT(blurred_image_expected.data[i], WithinAbs(blurred_datapack.data[i], tol));
    }

    // TODO: Apply your adjoint (deconvolution) and check that the output matches the this following image file
    vector<double> deblurred = ConvolutionOp.adjoint(blurred);
    DataPack<double> deblurred_datapack{true_image.width, true_image.height, deblurred};
    string deblurred_image_path = "test/data/images/TestSphere_deconvolved.pgm";
    DataPack unblurred_image_expected = ImageUtils::ReadImage<double>(deblurred_image_path);

    REQUIRE(deblurred_datapack.data.size() == unblurred_image_expected.data.size());

    // TODO: Check that your unblurred image is close to the expectation
    ImageUtils::WriteImage<double>(deblurred_datapack, "test/test_file/TestSphere_convolution/deconv.pgm");
    for (size_t i = 0; i < deblurred.size(); i++)
    {
        REQUIRE_THAT(unblurred_image_expected.data[i], WithinAbs(deblurred_datapack.data[i], tol));
    }
}

/****************************************************************************/
/* Section 3.2 - Gaussian Function Testing
/****************************************************************************/

// TODO: Write a test to check that your Gaussian function is working correctly
TEST_CASE("Test Gaussian Log-Likelihood", "Differentiable Function Tests")
{
    // TODO: Use small vectors of simple values for x and y so that you can calculate the expected result analytically
    vector<double> x = {1.0, 3.0, 5.0};
    vector<double> y = {1.0, 2.0, 2.0};
    double sigma = 1.0;

    // Check the result of a Gaussian function and its gradient with the Identity operator
    Identity<double> Id;
    double expected_Id = 5.0;
    vector<double> expected_gradient_Id = {0.0, 1.0, 3.0};

    GaussianLikelihood<double> gaussian_Id(Id, y, sigma);
    REQUIRE(gaussian_Id(x) == expected_Id);
    REQUIRE(gaussian_Id.gradient(x) == expected_gradient_Id);

    // Check the result of the Gaussian function and its gradient with a sub-sampling operator
    vector<size_t> I = {0, 2};
    vector<double> y_sub = {1.0, 2.0};
    SubSampling<double> sampler(I, 3);
    double expected_sampler = 4.5;
    vector<double> expected_gradient_sampler = {0.0, 0.0, 3.0};

    GaussianLikelihood<double> gaussian_sampler(sampler, y_sub, sigma);
    REQUIRE(gaussian_sampler(x) == expected_sampler);
    REQUIRE(gaussian_sampler.gradient(x) == expected_gradient_sampler);
}

/****************************************************************************/
/* Section 3.3 - DCT L1-Norm Testing
/****************************************************************************/

// TODO: Write a test to check that your DCT L1-Norm function is working correctly
TEST_CASE("Test DCT L1-Norm", "Non-Differentiable Function Tests")
{
    // Load the initial data
    string image_path = "test/data/images/TestSphere.pgm";
    DataPack image = ImageUtils::ReadImage<double>(image_path);

    int width = image.width;
    int height = image.height;
    vector<double> imag_data = image.data;

    // Define DCT l1-norm prior and apply it to the image to get l1_norm
    // Use default threshold tau = 0.1
    L1Norm<double> DCT_l1norm(width, height);
    double l1_norm = DCT_l1norm(imag_data);
    REQUIRE_THAT(l1_norm, WithinRel(1.93e8, 0.05));

    // Calculate the proximal and check that it is close to the expected proximal
    DataPack expected_prox = OptimisationUtils::ReadData<double>("test/data/measurements/TestSphere_proximal.dat");
    vector<double> prox_result = DCT_l1norm.proximal(imag_data);

    for (size_t i = 0; i < expected_prox.data.size(); i++)
    {
        REQUIRE_THAT(prox_result[i], WithinRel(expected_prox.data[i], 0.05));
    }
}

/****************************************************************************/
/* Section 3.5 - Integration Test
/****************************************************************************/

TEST_CASE("Recover From Synthetic Data", "Integration Test")
{
    // TODO: Write a test that checks a small optimisation problem using the opertors and functions
    // Load initial data
    double sigma = 0.1;
    double alpha = 0.01;
    double beta = 1e4;
    double tol = 0.01;
    int nmax = 50;

    string image_path = "test/data/images/TestSphere.pgm";
    DataPack image = ImageUtils::ReadImage<double>(image_path);
    size_t size = image.data.size();
    size_t size_I = size * 0.8;
    size_t size_drop = size - size_I;

    // Define your linear operator to drop 20% of the pixels in the image
    // Generates a random ordered indices array
    vector<size_t> indices_I;
    indices_I.reserve(size_I);

    std::unordered_set<int> indices_remove;
    std::random_device rd;
    std::mt19937 gen(rd());

    while (indices_remove.size() < size_drop)
    {
        int random_index = gen() % size;
        indices_remove.insert(random_index);
    }
    for (size_t i = 0; i < size; i++)
    {
        if (indices_remove.find(i) == indices_remove.end())
        {
            indices_I.push_back(i);
        }
    }

    REQUIRE(indices_I.size() == size_I);

    SubSampling<double> drop_operator(indices_I, size);
    vector<double> y = drop_operator(image.data);
    vector<double> dirty_image = drop_operator.adjoint(y);

    REQUIRE(dirty_image.size() == size);

    // Apply your optimisation (you may use the parameters above if you wish)
    // with the Gaussian likelihood and DCT l1-norm prior
    GaussianLikelihood<double> gaussian(drop_operator, y, sigma);
    L1Norm<double> dct_l1_norm(image.width, image.height);

    // Define a lambda function
    auto H_cost = [&gaussian, &dct_l1_norm](const vector<double>& x_prev, const vector<double>& x_curr, double tol) -> bool
    {
        return OptimisationUtils::ConvergedCost(x_prev, x_curr, gaussian, dct_l1_norm, tol);
    };

    vector<double> recovered_image = OptimisationUtils::IterativeAlgorithm<double, double>(
        gaussian,
        dct_l1_norm,
        y,
        drop_operator,
        alpha,
        beta,
        nmax,
        tol,
        H_cost);

    // Check that the result is close to the real image than the dirty image is
    // Calculate the diff between excted image data to the data of target image
    auto mse = [](const vector<double>& a, const vector<double>& b) -> double
    {
        double error = 0.0;
        for (size_t i = 0; i < a.size(); i++)
        {
            double diff = a[i] - b[i];
            error += diff * diff / a.size();
        }
        return error;
    };

    double mse_dirty = mse(image.data, dirty_image);
    double mse_recovered = mse(image.data, recovered_image);
    REQUIRE(mse_recovered < mse_dirty);

    // Write the dirty image for comparing
    DataPack<double> dirty_image_pack;
    dirty_image_pack.width = image.width;
    dirty_image_pack.height = image.height;
    dirty_image_pack.data = dirty_image;
    ImageUtils::WriteImage<double>(dirty_image_pack, "test/test_file/TestSphere_subsampling/dirty_imag.pgm");

    // Write the recoverd image for comparing
    DataPack<double> recovered_data_pack;
    recovered_data_pack.width = image.width;
    recovered_data_pack.height = image.height;
    recovered_data_pack.data = recovered_image;
    ImageUtils::WriteImage<double>(recovered_data_pack, "test/test_file/TestSphere_subsampling/recovered_imag.pgm");
}

// /****************************************************************************/
// /* Section 5.1 - FT Operator Testing
// /****************************************************************************/

TEST_CASE("Test FT", "FT Operator Testing")
{
    // Load initial data
    string image_path = "test/data/images/TestSphere.pgm";
    DataPack true_image = ImageUtils::ReadImage<double>(image_path);

    int width = true_image.width;
    int height = true_image.height;
    int N = width * height;
    vector<double> data = true_image.data;

    // Apply forward and inverse fourier transform
    FourierTransform ft(width, height);
    vector<complex<double>> freq_space = ft(data);
    vector<double> real_space = ft.adjoint(freq_space);

    // Check the FT agree with expected
    REQUIRE(real_space.size() == N);

    for (size_t i = 0; i < N; i++)
    {
        REQUIRE(std::abs(real_space[i] - data[i]) < 1e-4);
    }
}

// /****************************************************************************/
// /* Section 5.2 - Operator Composition Testing
// /****************************************************************************/

TEST_CASE("Test ComposedOperator with Identity", "[LinearOperator]")
{
    // Identity<double> Id;
    // Identity<double> phi;
    auto phi = std::make_shared<Identity<double>>();
    auto Id = std::make_shared<Identity<double>>();

    // Constructe the composed operator
    ComposedOperator<double, double, double> composition_1(phi, Id);
    ComposedOperator<double, double, double> composition_2(Id, phi);

    vector<double> x = {0.0, 1.0, 2.0};

    vector<double> c1 = composition_1(x);
    vector<double> c2 = composition_2(x);
    vector<double> p = (*phi)(x);

    // Check Phi o Id = Id o Phi = Phi
    REQUIRE(c1.size() == p.size());
    REQUIRE(c2.size() == p.size());
    for (size_t i = 0; i < p.size(); i++)
    {
        REQUIRE(c1[i] == c2[i]);
        REQUIRE(c2[i] == p[i]);
    }
}
