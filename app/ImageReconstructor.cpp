#include "CLI11.hpp"
#include "ImageUtils.h"
#include "LinearOperator.h"
#include "OptimisationUtils.h"
#include <filesystem>
#include <iostream>
#include <string>

using std::string;
using std::vector;
namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    string measurements_file;
    double alpha = 1.0;
    double beta = 1e4;
    double sigma = 1.0;
    double delta = 0.1;
    double eps = 0.1;
    double tau = 0.1;
    string kernel_size;
    string indices_file;

    // Add all flag option
    // note that for long string require two dashes
    CLI::App app{"App description"};
    app.add_option("-f", measurements_file, "The measurements data file")->required();
    app.add_option("--alpha", alpha, "The gradient step size")->capture_default_str();
    app.add_option("--beta", beta, "The proximal step size")->capture_default_str();
    app.add_option("--sigma", sigma, "The value of sigma in the log-likelihood function")->capture_default_str();
    app.add_option("--delta", delta, "The value of delta in the convergence function")->capture_default_str();
    app.add_option("-k", kernel_size, "The Gaussian kernel size in pixel only for convolution problem");
    app.add_option("-i", indices_file, "The indices file only for a subsampling problem");
    CLI11_PARSE(app, argc, argv);

    // use convolution problem or use subsampling problem
    bool useKernel = !kernel_size.empty();
    bool useIndices = !indices_file.empty();
    if (useKernel && useIndices)
    {
        throw std::runtime_error("Error: -k and -i are mutually exclusive \n");
    }
    else if (!useKernel && !useIndices)
    {
        throw std::runtime_error("Error: must provide either -k or -i \n");
    }

    DataPack y_data_pack = OptimisationUtils::ReadData<double>(measurements_file);
    std::unique_ptr<LinearOperator<double, double>> op;

    size_t w = y_data_pack.width;
    size_t h = y_data_pack.height;
    size_t N = w * h;

    if (useKernel)
    {
        vector<double> kernel = ImageUtils::GenSincKernel(w, h, std::stod(kernel_size));
        op = std::make_unique<Convolution>(kernel, w, h, eps);
    }
    else
    {
        DataPack idx_pack = OptimisationUtils::ReadData<double>(indices_file);
        vector<size_t> indices(idx_pack.data.size());
        for (size_t i = 0; i < idx_pack.data.size(); i++)
        {
            indices[i] = idx_pack.data[i];
        }
        op = std::make_unique<SubSampling<double>>(indices, N);
    }

    // save dirty image
    vector<double> dirty = (*op).adjoint(y_data_pack.data);
    DataPack<double> dirty_pack{w, h, dirty};
    string dirty_file = "images/" + fs::path(measurements_file).stem().string() + "_dirty.pgm";
    ImageUtils::WriteImage<double>(dirty_pack, dirty_file);

    // Iteration application
    GaussianLikelihood<double> f(*op, y_data_pack.data, sigma);
    L1Norm<double> g(w, h, tau);
    vector<double> x = dirty;
    size_t nmax = 500;

    auto H_cost = [&f, &g](const vector<double>& x_prev, const vector<double>& x_curr, double tol) -> bool
    {
        return OptimisationUtils::ConvergedCost(x_prev, x_curr, f, g, tol);
    };

    vector<double> recovered_image = OptimisationUtils::IterativeAlgorithm<double, double>(
        f,
        g,
        y_data_pack.data,
        *op,
        alpha,
        beta,
        nmax,
        delta,
        H_cost);

    // save recovered image
    DataPack<double> recovered_pack{w, h, recovered_image};
    string recovered_file = "images/" + fs::path(measurements_file).stem().string() + "_reconstructed.pgm";
    ImageUtils::WriteImage<double>(recovered_pack, recovered_file);
    return 0;
}