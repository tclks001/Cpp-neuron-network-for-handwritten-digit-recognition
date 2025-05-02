// layer class `Layer` which is a class for a single layer of neurons in a neural network.

# include "matrix.cpp"
# include <iostream>
# include <vector>
# include <cmath>
# include <algorithm>
# include <random>
using namespace std;

class Layer {
public:
    Matrix<double> W; // weight
    Matrix<double> b; // bias
    Matrix<double> dW; // weight gradient
    Matrix<double> db; // bias gradient
    Matrix<double> z; // pre-activation
    Matrix<double> a; // activation
    Matrix<double> dz; // difference
    enum activation {
        sigmoid,
        relu,
        softmax,
    };
    activation act_type; // activation function type
    int num_neurons; // number of neurons in the layer
    Layer(int num_inputs, int batch_size, int num_neurons, activation act_type) {
        this->num_neurons = num_neurons;
        this->act_type = act_type;

        // initialize weight matrix and bias vector. weights based on He or Xavier initialization
        W = Matrix<double>(
            num_inputs, 
            num_neurons, 
            act_type == relu ? sqrt(2.0/num_inputs) : sqrt(2.0/(num_inputs+num_neurons))
        );
        b = Matrix<double>(1, num_neurons);
        z = Matrix<double>(batch_size, num_neurons);
        a = Matrix<double>(batch_size, num_neurons);
        dz = Matrix<double>(batch_size, num_neurons);
        dW = Matrix<double>(num_inputs, num_neurons);
        db = Matrix<double>(1, num_neurons);
    }

    // forward propagation from input matrix X to output matrix a
    Matrix<double> forward(Matrix<double>& X) {
        z = W * X + b; // pre-activation matrix
        if(act_type == sigmoid) {
            a = z.sigmoid(); // sigmoid activation function
        }else if(act_type == relu) {
            a = z.relu(); // relu activation function
        }else if(act_type == softmax) {
            a = z.softmax(); // softmax activation function
        }
        return a; // return activation matrix
    }

    
    Matrix<double> backward(Matrix<double>& dA, Matrix<double>& A_prev) {
        // dz
        if(act_type == sigmoid) {
            dz = dA * (z.sigmoid_deriv()); // derivative of sigmoid function
        }else if(act_type == relu) {
            dz = dA * (z.relu_deriv()); // derivative of relu function
        }else if(act_type == softmax) {
            dz = dA * (z.softmax_deriv()); // derivative of softmax function
        }

        int m = A_prev.rows;

        // dW
        dW = (A_prev.transpose() * dz) * (1.0/m) ;

        // db
        int B = A_prev.cols;
        Matrix<double> ones(B, 1);
        for(int i=0; i<B; i++){
            ones.data[i][0] = 1.0/B;
        }
        db = dz * ones;

        // dA_prev
        Matrix<double> dA_prev = dz * W;
        return dA_prev; // return dA_prev matrix
    }

    void update(Matrix<double>& delta, double learning_rate) {
        W = W + dW * learning_rate * -1.0;
        b = b + db * learning_rate * -1.0;
    }

};