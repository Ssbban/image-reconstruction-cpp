add_library(OptimisationLib STATIC OptimisationUtils.cpp ImageUtils.cpp LinearOperator.cpp DifferentiableFunction.cpp NonDifferentiableFunction.cpp)
target_link_libraries(OptimisationLib PUBLIC FFTW3::fftw3)
target_include_directories(OptimisationLib PUBLIC "${CMAKE_SOURCE_DIR}/Optimisation/include")