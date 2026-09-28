# Respones
This file is used to respond to all questions in the assignment.


## Part 2.3: Limitations of this test
- In real case, different initial values ​​may lead to different convergence paths or fall into local extreme values. In this test case, `f(x)` is a qudratic function, while the initial value is simply set to `y = 1`, and the problem itself has only one global minimum point 0. This test method is too simple to test what happens when more complex functions are introduced. In real cases, there may be multiple local minima in a function. 

- `g(x)` is empty in this case, so the proximal operation will not contribute to the iteration process. This makes it impossible to confirm whether this code handles non-differentiable functions correctly.

- The `Phi(x)=x` used in the test is the identity operator, so it does not change the input function. This is the simplest linear operator, and it is impossible to test the case where the linear operator maps a double to a complex.


## Part 4.3: Convergence Functions
The Convergence function determines whether the iteration algorithm has met the stop condition, so it is necessary to ensure that the convergence function can be correctly passed into the optimization algorithm.

This base converged function is given that `\[ \frac{|x_{n+1} - x_n|}{|x_n|} \]`, and the new coverged function is `\[ H(x_n, x_{n+1}, f, g, \delta) = \frac{|C(x_{n+1}) - C(x_n)|}{C(x_n)} < \delta \]`. 

### The type of parameter
The based convergence function only takes three arguments and has a signature with,

```h
bool converged(const vector<T>& x_prev, const vector<T>& x_next, double tol)
```

while the user also want to add a new convergence function using the cost function to judge the convergence criteria, thus it takes five arguments with signature as,

```h
bool ConvergedCost(
    const vector<T>& x_prev,
    const vector<T>& x_next,
    const DifferentiableFunction<T>& f,
    const NonDifferentiableFunction<T>& g,
    double tol)
```

To allow flexibility in choosing the convergence function in the algorithm, we need to ensure that the converged function signature is compatible with the expected type in `IterativeAlgorithm`, which means the function should accept either three arguments or five arguments as shown above,

- two vectors `x_prev` and `x_next`.
- `DifferentiableFunction` and `NonDifferentiableFunction`
- tolerance value `delta`.

It should then return a `bool` type indicating whether convergence has been achieved. In C++, function templates cannot be passed directly as default parameters but can be wrapped using function pointers or lambda expressions. Therefore, it is very important to pass the convergence function correctly.

### Previous used method (optional)
The simplest way is to explicitly write out the function overloads for these two situations. This is the most intuitive method and the one I used at the beginning. Although it is not flexible, I still list this method,

```cpp
// My previous method
template <typename Tx, typename Ty>
vector<Tx> IterativeAlgorithm(
    const DifferentiableFunction<Tx>& f,
    const NonDifferentiableFunction<Tx>& g,
    const vector<Ty>& y,
    const LinearOperator<Tx, Ty>& phi,
    double alpha,
    double beta,
    size_t n_max,
    double delta,
    bool (*H_cost)(const vector<Tx>&, const vector<Tx>&, const DifferentiableFunction<Tx>&, const NonDifferentiableFunction<Tx>&, double))
```

### Compatibility
A more flexible method is used in this assignment now, which is using the lambda function to pass them. The `IterativeAlgorithm` can accept, a vector-based default function `converged` or a custom cost-based converged function `ConvergedCost`. To ensure type correctness, I modified the `converged` function to accept `const vector<T>&` instead of `vector<T>&`.

### Explicit code:
The `std::function` helps me to build the lambda function, if the user need to pass a non-default converged function `ConvergedCost`, run the below code,

```cpp
auto H_cost = [&f, &g](const vector<double>& x_prev, const vector<double>& x_curr, double tol) -> bool
{
    return OptimisationUtils::ConvergedCost(x_prev, x_curr, f, g, tol);
};

vector<double> recovered_image = IterativeAlgorithm<double, double>(
    f,
    g,
    measurements,
    operator,
    alpha,
    beta,
    n_max,
    tol,
    H_cost);
```

## Part 5: ImageReconstructor simple test

Both of the examples here use the custom converge function `ConvergedCost` with `n_max = 500`.

### Sub-sampling Problem:
The subsampling example codes used for the results `images/UtahTeapot_subSampledMeasurements_dirty.pgm` and `images/UtahTeapot_subSampledMeasurements_reconstructed.pgm` are given below,

```bash
./build/bin/ImageReconstructor -f data/measurements/UtahTeapot_subSampledMeasurements.dat -i data/operators/UtahTeapot_sampledPixels.dat --alpha 1.0 --beta 1e4 --sigma 1.0 --delta 0.005
```

### Convolution Problem:
The convolution example code used for the results `images/UtahTeapot_convolved_dirty.pgm` and `images/UtahTeapot_convolved_reconstructed.pgm` is given below, 

```bash
./build/bin/ImageReconstructor -f data/measurements/UtahTeapot_convolved.dat -k 1.0 --alpha 1.0 --beta 1e4 --sigma 1.0 --delta 0.1
```

### Command:
The parameters using here are,

```
-h: Return help message and exit.

-f: Path to the convolved measurement data file, which should be a string type (Require para).

--alpha: The gradient step size, which should be float type (Default = 1.0).

--beta: The proximal step size, which should be float type (Default = 10000.0).

--sigma: The parameter in the log-likelihood function, which should be float type (Default = 1.0).

--delta: The tolerance for cost-based convergence, which should be float type (Default = 0.1).

-k: The Gaussian kernel size in pixel, given only for a convolution problem, which should be float type (Exclusive with flag -i).

-i: The indices file, is given only for a subsampling problem, which should be a string type (Exclusive with flag -k).
```

In addition, the implicit parameters used in the algorithm are `epsilon = 0.1` and `tau = 0.1` respectively.

## Part 6.2: Composition Operator
### Type correctness
Our composition operator has three type parameters `T_in`, `T_mid` and `T_out`, which respect to the input type, mid type and output type. This operator has two main function, `operator()`, and the `adjoint()`,

Noticed that the two Linear Operators `\Phi` and `\Psi` are composed together, while the former map `T_in` to `T_mid` and the other map `T_mid` to `T_out`, which means,
- `T_in`: Input type of the first operator phi 
- `T_mid`: Output of phi and input of psi
- `T_out`: Final output of phi

The structure of the code is,

```cpp
template <typename T_in, typename T_mid, typename T_out>
class ComposedOperator : public LinearOperator<T_in, T_out>
{
public:
    ComposedOperator(shared_ptr<LinearOperator<T_in, T_mid>> op1, shared_ptr<LinearOperator<T_mid, T_out>> op2) : phi(op1), psi(op2){}

    vector<T_out> operator()(const vector<T_in>& x) const override{}
    vector<T_in> adjoint(const vector<T_out>& y) const override{}
};
```

Thus, the `operator()` map first uses `\Phi` to get `vector<T_mid>`, then uses `\Psi` to get `vector<T_out>`, while the `adjoint()` does the inverse mapping process. In this assignment, `T_in` must be `double`. 

I previously use explicit template to make sure the type correction as shown in the below code block.

```cpp
template class ComposedOperator<double, double, double>;
template class ComposedOperator<double, double, complex<double>>;
template class ComposedOperator<double, complex<double>, double>;
template class ComposedOperator<double, complex<double>, complex<double>>;
```
However, this code is not efficiency enough, thus, this assignment using method based on shared pointers with implicit template. Because this template requires that the output type of first operator is the same as the input of the second operator while use smart pointers to store references to these operators, it automatically manages lifetimes, so as long as the user respect the correct type parameters the code will compile and run safely.

### Data copying, sharing, and ownership
Each `ComposedOperator` object contains `std::shared_ptr<LinearOperator<T_in, T_mid>> op1;` and `std::shared_ptr<LinearOperator<T_mid, T_out>> op2`. In other words, we share operator instances instead of copying them. In each operator call, we usually care about the result, and by returning it by value or storing it in a local vector, we avoid complex ownership issues (this is also the reason why my code took a long time to run before, and it was called repeatedly when running `gradient`, doing a lot of meaningless calculations). Another benefit of using this method is that we don't have to track the lifetime manually, once nothing else refers to these operators, they will be automatically released. Therefore, the code does not force a single lifetime policy, and the operator will remain valid as long as any reference remains in scope.

## Part 7 (Optional)
The frequency reconstructor example code respect to `images/UtahTeapot_frequencyLossMeasurements_dirty.pgm` and `images/UtahTeapot_frequencyLossMeasurements_reconstructed.pgm` is given below, 

```bash
./build/bin/FrequencyReconstructor -f data/measurements/UtahTeapot_frequencyLossMeasurements.dat -i data/operators/UtahTeapot_sampledFrequencies.dat --alpha 1.0 --beta 1e4 --sigma 1.0 --delta 0.1
```