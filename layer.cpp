// layer class `Layer` which is a class for a single layer of neurons in a neural network.

# include "matrix.cpp"
# include <iostream>
# include <vector>
# include <cmath>
# include <algorithm>
using namespace std;

class Layer {
public:
    Matrix<double> W; // weight matrix
    Matrix<double> b; // bias vector
    Matrix<double> z; // pre-activation matrix
    Matrix<double> a; // activation matrix
    enum activation {
        sigmoid,
        relu,
        softmax,
    };
    activation act_type; // activation function type
    
};