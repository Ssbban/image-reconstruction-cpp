#pragma once
#include "DataPack.h"
#include "DifferentiableFunction.h"
#include "LinearOperator.h"
#include "NonDifferentiableFunction.h"
#include <cmath>
#include <fstream>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

using std::invalid_argument;
using std::out_of_range;
using std::runtime_error;
using std::string;
using std::vector;

namespace OptimisationUtils
{
/**
 * @brief Utility function for splitting a string into a list of sub-strings separated by delimiter
 *
 * @param line
 * @param delim
 * @return vector<string>
 */
vector<string> tokenise(const string& line, char delim = ' ');

/**
 * @brief Utility function for converting list of string tokens to a data type T.
 * Must be explicitly instantiated in OptimisationUtils.cpp due to differences in implementation.
 *
 * @tparam T
 * @param tokens
 * @return T
 */
template <typename T>
T getDataFromTokens(const vector<string>& tokens);

/**
 * @brief Read a data file. The width and height of the related image as stored
 * in the reference parameters. Requires getDataFromTokens to be implements for
 * type T.
 *
 * @tparam T
 * @param path
 * @param width
 * @param height
 * @return vector<T>
 */
template <typename T>
DataPack<T> ReadData(const string& path)
{
    std::fstream file;
    file.open(path, std::ios::in);

    // Check the file is exsist
    if (!file.is_open())
    {
        throw runtime_error("ERROR: File is not found in " + path);
    };
    string line;
    if (!std::getline(file, line))
    {
        throw runtime_error("ERROR: The file is empty");
    }

    // Check the dimension of image
    vector<string> tokens = tokenise(line);
    if (tokens.size() != 2)
    {
        throw invalid_argument("ERROR: The dimension of the image should be 2");
    }

    size_t width, height;
    try
    {
        width = std::stoi(tokens[0]);
        height = std::stoi(tokens[1]);
    }
    catch (const invalid_argument& e)
    {
        throw invalid_argument("ERROR: The dimensions of images must be an integer in");
    }
    catch (const out_of_range& e)
    {
        throw out_of_range("ERROR: The dimensions of images is out of range");
    }

    if (width < 0 || height < 0)
    {
        throw invalid_argument("ERROR: The width and height can not be negative");
    }

    // Check the data size of image
    if (!std::getline(file, line))
    {
        throw runtime_error("ERROR: Missing the number of data size ");
    }

    tokens = tokenise(line);
    if (tokens.size() != 1)
    {
        throw invalid_argument("ERROR: The number of data should be only one number ");
    }

    size_t data_size;
    try
    {
        data_size = std::stoi(tokens[0]);
    }
    catch (const invalid_argument& e)
    {
        throw invalid_argument("ERROR: Expected the number of data are integer ");
    }
    catch (const out_of_range& e)
    {
        throw out_of_range("ERROR: The total number of data is out of range ");
    }

    if (data_size < 0)
    {
        throw runtime_error("ERROR: Invalid negative data size");
    }

    // Check the data of the file
    vector<T> data(data_size);
    size_t count = 0;
    while (std::getline(file, line))
    {
        vector<string> tokens = tokenise(line);
        try
        {
            data[count++] = getDataFromTokens<T>(tokens);
        }
        catch (const out_of_range& e)
        {
            throw runtime_error("ERROR: The data value is out of range");
        }
        catch (const invalid_argument& e)
        {
            throw runtime_error("ERROR: Failed to parse data at line " + std::to_string(count + 2));
        }
    }
    file.close();
    return {width, height, data};
}

/**
 * @brief Write a data file
 *
 * @tparam Datatype T must overload the << operator for streaming to file.
 * @param data
 * @param width
 * @param height
 * @param path
 */
template <typename T>
void WriteData(const vector<T>& data, size_t width, size_t height,
               const string& path)
{
    std::fstream file;
    file.open(path, std::ios::out);
    file << width << " " << height << "\n";
    file << data.size() << "\n";

    for (size_t i = 0; i < data.size(); i++)
    {
        file << data[i] << "\n";
    }
    file.close();
}

/**
 * @brief Simple convergence function that checks the relative change in two
 * images
 *
 * The operators * and - must be implemented for the data type T
 * The result of the squared difference should be able to be added to the double
 * type.
 *
 * @param x1 : vector of data
 * @param x2 : updated vector of data
 * @param tol : tolerance for convergence
 * @return true
 * @return false
 */
template <typename T>
bool converged(const vector<T>& x1, const vector<T>& x2, double tol)
{
    double sum = 0;
    for (size_t i = 0; i < x1.size(); i++)
    {
        sum += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    }
    double ms_diff = sqrt(sum / x1.size());
    return ms_diff <= tol;
}

/**
 * @brief Checks customised convergence based on relative change in cost function.
 *
 * @param x_prev Previous iteration.
 * @param x_next Current iteration.
 * @param f Differentiable function (gradient-based term).
 * @param g Non-differentiable function (proximal term).
 * @param tol Convergence threshold.
 * @return True if relative cost change is within tolerance.
 * @throws runtime_error If cost function value is too close to zero.
 */
template <typename T>
bool ConvergedCost(const vector<T>& x_prev, const vector<T>& x_next,
                   const DifferentiableFunction<T>& f,
                   const NonDifferentiableFunction<T>& g, double tol)
{
    double C_prev = f(x_prev) + g(x_prev);

    // Check the C_prev not close to zero
    if (std::abs(C_prev) < 1e-10)
    {
        throw runtime_error("ERROR: Cost function too close to zero, potential division error");
    }

    double C_next = f(x_next) + g(x_next);
    double relative_change = std::abs(C_next - C_prev) / C_prev;
    return (relative_change < tol);
}

/**
 * @brief Solves an optimization problem using an iterative approach.
 * This algorithm alternates between a gradient step and a proximal step.
 * @tparam Tx : Data type of optimization variable
 * @tparam Ty : Data type of measurement set
 * @param f : Differentiable function
 * @param g : Non-differentiable function
 * @param y : Measurement set
 * @param phi : Linear operator mapping x to y
 * @param alpha : Step size for gradient descent
 * @param beta : Step size for proximal update
 * @param n_max : Maximum number of iterations
 * @param delta : Convergence threshold
 * @param H : Convergence check function (default as `converged`).
 * @return The optimized variable x
 */
template <typename Tx, typename Ty>
vector<Tx> IterativeAlgorithm(
    const DifferentiableFunction<Tx>& f, const NonDifferentiableFunction<Tx>& g,
    const vector<Ty>& y, const LinearOperator<Tx, Ty>& phi, double alpha,
    double beta, size_t n_max, double delta,
    std::function<bool(const vector<Tx>&, const vector<Tx>&, double)> H =
        [](const vector<Tx>& x1, const vector<Tx>& x2, double tol)
    {
        return converged<Tx>(x1, x2, tol);
    })

{
    // Load initial data
    vector<Tx> x = phi.adjoint(y);
    vector<Tx> x_next(x.size());

    for (size_t n = 0; n < n_max; n++)
    {
        // Equation find x_prime
        vector<Tx> gradient_f = f.gradient(x);
        for (size_t i = 0; i < x.size(); i++)
        {
            x_next[i] = x[i] - alpha * gradient_f[i];
        }

        // Equation find x_{n+1}
        vector<Tx> proximal_g = g.proximal(x_next);
        for (size_t i = 0; i < x.size(); i++)
        {
            x_next[i] = x_next[i] - beta * proximal_g[i];
        }

        if (H(x, x_next, delta))
        {
            return x_next;
        }

        x = x_next;
    }
    return x;
}
} // namespace OptimisationUtils