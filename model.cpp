# include "matrix.cpp"
# include "layer.cpp"
# include <iostream>
# include <vector>
# include <cmath>
# include <algorithm>
# include <random>
using namespace std;
class Model {
public:
    vector<Layer> layers;
    enum loss_type { cross_entropy, mean_squared_error };
    void add_layer(Layer layer) {
        layers.push_back(layer);
    }
    Matrix<double> forward(Matrix<double>& input) {
        for (int i = 0; i < layers.size(); i++) {
            input = layers[i].forward(input);
        }
        return input;
    }
    void backward(Matrix<double>& x, Matrix<double>& y){
        Matrix<double> da = layers.back().a + y * -1;
        for(int i = layers.size()-1; i >= 0; --i){
            Matrix<double> A_prev = (i == 0) ? x : layers[i-1].a;
            da = layers[i].backward(da, A_prev);
        }
    }
    void train(Matrix<double>& x, Matrix<double>& y, int epochs, int batch_size, double learning_rate) {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<int> dist(0, 1);
        for (int i = 0; i < epochs; ++i) {
            vector<Matrix<double>> batchs_x, batchs_y;
            
        }
    }
};

