#include "CLI11.hpp"
#include "ImageUtils.h"
#include "OptimisationUtils.h"
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    string measurements_file;
    double alpha = 1.0;
    double beta = 1e4;
    double sigma = 1.0;
    double delta = 0.1;
    double tau = 1.0;
    string indices_file;

    // parse all argument and set flag
    CLI::App app{"App description"};
    app.add_option("-f", measurements_file, "The measurements data file")->required();
    app.add_option("-i", indices_file, "The indices file only for a subsampling problem")->required();
    app.add_option("--alpha", alpha, "The gradient step size")->capture_default_str();
    app.add_option("--beta", beta, "The proximal step size")->capture_default_str();
    app.add_option("--sigma", sigma, "The value of sigma in the log-likelihood function")->capture_default_str();
    app.add_option("--delta", delta, "The value of delta in the convergence function")->capture_default_str();
    CLI11_PARSE(app, argc, argv);

    // Read the input file
    DataPack y_datapack = OptimisationUtils::ReadData<complex<double>>(measurements_file);
    size_t w = y_datapack.width;
    size_t h = y_datapack.height;

    DataPack idx_pack = OptimisationUtils::ReadData<double>(indices_file);
    vector<size_t> idx(idx_pack.data.size());
    for (size_t i = 0; i < idx_pack.data.size(); i++)
    {
        idx[i] = idx_pack.data[i];
    }

    // Construct operator
    auto ft = std::make_shared<FourierTransform>(w, h);
    auto sub_op = std::make_shared<SubSampling<complex<double>>>(idx, w * h);
    ComposedOperator<double, complex<double>, complex<double>> freq_op(ft, sub_op);

    // Determine dirty
    vector<double> dirty = freq_op.adjoint(y_datapack.data);
    DataPack<double> dirty_pack{w, h, dirty};
    string dirty_file = "images/" + fs::path(measurements_file).stem().string() + "_dirty.pgm";
    ImageUtils::WriteImage<double>(dirty_pack, dirty_file);

    // Iteration application
    GaussianLikelihood<complex<double>> f(freq_op, y_datapack.data, sigma);
    L1Norm<double> g(w, h, tau);
    size_t nmax = 500;
    auto H_cost = [&f, &g](const vector<double>& x_prev, const vector<double>& x_curr, double tol) -> bool
    {
        return OptimisationUtils::ConvergedCost(x_prev, x_curr, f, g, tol);
    };
    vector<double> recovered = OptimisationUtils::IterativeAlgorithm<double, complex<double>>(
        f,
        g,
        y_datapack.data,
        freq_op,
        alpha,
        beta,
        nmax,
        delta,
        H_cost);

    DataPack<double> recon_pack{w, h, recovered};
    string recon_file = "images/" + fs::path(measurements_file).stem().string() + "_reconstructed.pgm";
    ImageUtils::WriteImage<double>(recon_pack, recon_file);

    return 0;
}