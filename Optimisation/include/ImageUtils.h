#pragma once

#include <DataPack.h>
#include <string>
#include <vector>

using std::string;
using std::vector;

namespace ImageUtils
{
/**
 * @brief Read an image file from .pgm format.
 * Reference parameters width and height are overwritten with information from the file.
 * This should be replace with a single output of the Image class once written.
 *
 * @param path
 * @param width
 * @param height
 * @return vector<double>
 */
template <typename T>
DataPack<T> ReadImage(const string& path);

/**
 * @brief Write an image to file in .pgm format
 * You can pass separate width, height, and image data variables as an DataPack
 * struct using the syntax {width, height, data}.
 * @param image
 * @param path
 * @return * void
 */
template <typename T>
void WriteImage(const DataPack<T>& image, const string& path);

/**
 * @brief Generate a Gaussian kernel of a given size
 *
 */
vector<double> GenSincKernel(size_t w, size_t h, double scale);
} // namespace ImageUtils

// extern template class DataPack<double>;