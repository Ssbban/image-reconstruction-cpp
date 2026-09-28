#include "OptimisationUtils.h"
#include <complex>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using std::complex;
using std::string;
using std::vector;

vector<string> OptimisationUtils::tokenise(const string& line, char delim)
{
    vector<string> tokens;
    size_t pos = 0;
    size_t next = 0;
    do
    {
        next = line.find(delim, pos);
        tokens.push_back(line.substr(pos, (next - pos)));
        pos = next + 1;
    } while (next != string::npos);

    return tokens;
}

template <>
double OptimisationUtils::getDataFromTokens<double>(const vector<string>& tokens)
{
    if (tokens.size() != 1)
    {
        throw std::invalid_argument("ERROR: Expected one value for real number");
    }
    return std::stod(tokens[0]);
}

template <>
complex<double> OptimisationUtils::getDataFromTokens<complex<double>>(const vector<string>& tokens)
{
    if (tokens.size() != 2)
    {
        throw std::invalid_argument("ERROR: Expected two values for complex number");
    }
    try
    {
        double real = std::stod(tokens[0]);
        double imag = std::stod(tokens[1]);
        return complex<double>(real, imag);
    }
    catch (const std::invalid_argument& e)
    {
        throw std::invalid_argument("ERROR: Invalid complex number format ");
    }
}