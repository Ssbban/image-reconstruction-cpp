# Image Reconstruction

## Assignment 1 Intro
This repository contains Assignment 1 of the C++ module at UCL, which uses the C++ language. The main goal of this project is to achieve image reconstruction, and some examples are given in this document. This project implements a general optimization method by building differentiable, non-differentiable functions and linear operators and uses the FFTW library for Fourier and DCT transforms. Applications include deblurring, subsampling reconstruction, and frequency space recovery.

This project is compiled using CMake, requiring at least CMake version 3.28 and C++17.

The main function is,

### I/O:
Supports reading and writing of data in `.dat` format, including `double` and `std::complex<double>`, and also supports reading and writing of `.pgm `images.

### Library Code:
- `LinearOperator` and its sub-class, including `Identity`, `SubSampling`, `Convolution`, `FourierTransform`, and `ComposedOperator`.
- `DifferentiableFunction`, such as `Quadratic` and `GaussianLikelihood`.
- `NonDifferentiableFunction`, such as `Empty` and `L1Norm`.
- `OptimisationUtils`, including `IterativeAlgorithm` and `ConvergedCost`, etc.

### Application:
This assignment gives some examples using the application and puts the example in `image` file.

`ImageReconstructor` will reconstruct for "subsampled" or "convolved" measurements. 
`FrequencyReconstructor` will reconstruct the image for frequency domain sampling scenarios。

### Project structure
```
COMPO210Assignment1/
├── app/                # Application code (ImageReconstructor, FrequencyReconstructor)
├── external/           # The external lib include CLI11
├── Optimisation/       # Library code (functions, operators, algorithms)
├── test/               # Unit tests
├── data/               # Input datasets and measurement files
├── images/             # Output images in .pgm format
├── CMakeLists.txt      # Main build configuration
├── README.md           # This file
└── Responses.md        # Response to some question in the note
```

## Dependence
This repository requires,
- C++17 or higher compiler.
- CMake 3.28+ for building.
- FFTW3 for implementing FFT.
- Catch2 for unit testing.
- CLI11 for parse command and I already put it in the `external` file. Users can download it at: https://github.com/CLIUtils/CLI11 if needed.

### Catch2:
Catch 2 is require for the unit test, run the code below to install,
```bash
git clone https://github.com/catchorg/Catch2 Catch2
cd Catch2
cmake -B build -DBUILD_TESTING=OFF
cmake --build build
cmake --install build/.
```

### FFTW3:
Run the code below to install the global environment,

```bash
wget http://fftw.org/fftw-3.3.10.tar.gz
tar -xvzf fftw-3.3.10.tar.gz
cd fftw-3.3.10
mkdir build && cd build
cmake .. 
make -j
sudo make install
```
## Compile and build
In the same directory as this README.md file, run the following code in the terminal to build and compile,

```bash
cmake -B build
cmake --build build
```

Then you should now be able to find `ImageReconstructor`, `FrequencyReconstructor` and `TestOptimisation` in the `/build/bin/` folder. To test this program, run `./build/bin/PROGRAM_NAME` in the terminal. The `TestOptimisation` contains only the unit test. All tests should be passed in normal situations when running `./build/bin/TestOptimisation` in terminal.

## Application

### ImageReconstructor
This application is designed to perform image reconstruction for "subsampling" or "convolution" problems. The user needs to provide a dataset that has been "subsampled" or blurred by "convolution" and reconstruct the image using a pre-written iterative optimization. The iterative algorithm is based on gradient descent and uses a Gaussian log-likelihood for `f` and a DCT L1-Norm for `g`. The user can provide different parameters or use the default parameters, but the measurements data file and the Gaussian kernel size in pixel for convolution / the indices file for subsampling are necessary.

The below code explains how to implement the sub-sampling problem, the dirty and reconstructed images of the data will be output to the `image` folder as `FILENAME_dirty.pgm` and `FILENAME_reconstructed.pgm` respectively.

```bash
# Sub-sampling
./build/bin/ImageReconstructor \
    -f data/measurements/UtahTeapot_subSampledMeasurements.dat \
    -i data/operators/UtahTeapot_sampledPixels.dat \
    --alpha 1.0 \
    --beta 1e4 \
    --sigma 1.0 \
    --delta 0.005 \
```
For convolution,
```bash
# Convolution
./build/bin/ImageReconstructor \
    -f data/measurements/UtahTeapot_convolved.dat \
    -k 1.0 \
    --alpha 1.0 \
    --beta 1e4 \
    --sigma 1.0 \
    --delta 0.1
```

### FrequencyReconstructor
The application of `ImageReconstructor` takes the sampled data in real space as input, but the measurement `y` can be the image sampled in the Fourier domain. Therefore, this part assumes that the measurement information comes from the frequency coverage in Fourier space, and then we can combine the measurement Fourier transform and subsampling operator as a new measurement operator by operator composition, to reconstruct the original image by the same iterative optimization method.

Run the below code to run the FrequencyReconstructor,
```bash
./build/bin/FrequencyReconstructor \
    -f data/measurements/UtahTeapot_frequencyLossMeasurements.dat \
    -i data/operators/UtahTeapot_sampledFrequencies.dat \
    --alpha 1.0 \
    --beta 1e4 \
    --sigma 1.0 \
    --delta 0.5
```
