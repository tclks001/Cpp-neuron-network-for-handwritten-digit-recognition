#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

using namespace std;

template <typename T>
class Matrix {
public:
    int rows, cols;
    vector<vector<T>> data;
    Matrix() : rows(0), cols(0), data() {}
    Matrix(int r, int c) : rows(r), cols(c), data(r, vector<T>(c)) {}

    // constructor to initialize weight matrix with random values based on normal distribution
    Matrix(int r,int c, double stddev){
        rows = r;
        cols = c;
        data = vector<vector<T>>(r, vector<T>(c));
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> dist(0.0, stddev);
        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                data[i][j] = dist(gen); // fill weight matrix with random values based on normal distribution
            }
        }
    }
    Matrix(const vector<vector<T>>& d) : rows(d.size()), cols(d[0].size()), data(d) {}
    Matrix(const Matrix<T>& m) : rows(m.rows), cols(m.cols), data(m.data) {}
    ~Matrix() {}
    Matrix<T>& operator=(const Matrix<T>& m) {
        if (this!= &m) {
            rows = m.rows;
            cols = m.cols;
            data = m.data;
        }
        return *this;
    }
    Matrix<T> operator+(const Matrix<T>& m) const {
        Matrix<T> res(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res.data[i][j] = data[i][j] + m.data[i][j];
            }
        }
        return res;
    }
    Matrix<T> operator*(const Matrix<T>& m) const {
        Matrix<T> res(rows, m.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                for (int k = 0; k < cols; k++) {
                    res.data[i][j] += data[i][k] * m.data[k][j];
                }
            }
        }
        return res;
    }
    Matrix<T> operator*(const T& x) const {
        Matrix<T> res(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res.data[i][j] = data[i][j] * x;
            }
        }
        return res;
    }
    Matrix<T> transpose() const {
        Matrix<T> res(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res.data[j][i] = data[i][j];
            }
        }
        return res;
    }

    // activation sigmoid
    Matrix<T> sigmoid() const {
        Matrix<T> res(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res.data[i][j] = 1 / (1 + exp(-data[i][j]));
            }
        }
        return res;
    }

    // activation relu
    Matrix<T> relu () const {
        Matrix<T> res(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res.data[i][j] = max(0.0, data[i][j]);
            }
        }
        return res;
    }

    // activation softmax
    Matrix<T> softmax() const {
        Matrix<T> res(rows, cols);
        for (int i = 0; i < rows; i++) {
            double sum = 0;
            for (int j = 0; j < cols; j++) {
                sum += exp(data[i][j]);
            }
            for (int j = 0; j < cols; j++) {
                res.data[i][j] = exp(data[i][j]) / sum;
            }
        }
        return res;
    }

    // derivative of sigmoid
    Matrix<T> sigmoid_deriv() const {
        Matrix<T> res(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res.data[i][j] = data[i][j] * (1 - data[i][j]);
            }
        }
        return res;
    }

    // derivative of relu
    Matrix<T> relu_deriv() const {
        Matrix<T> res(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res.data[i][j] = data[i][j] > 0 ? 1 : 0;
            }
        }
        return res;
    }

    // derivative of softmax
    Matrix<T> softmax_deriv() const {
        Matrix<T> res(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res.data[i][j] = data[i][j] * (1 - data[i][j]);
            }
        }
        return res;
    }

    void print() const {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cout << data[i][j] << " ";
            }
            cout << endl;
        }
    }
};

int main() {
    Matrix<double> m1(2, 3);
    m1.data[0][0] = 1;
    m1.data[0][1] = 2;
    m1.data[0][2] = 3;
    m1.data[1][0] = 4;
    m1.data[1][1] = 5;
    m1.data[1][2] = 6;
    /*
    [[1,2,3],
     [4,5,6]]
    */
    m1.print();
    Matrix<double> m2(3, 2);
    m2.data[0][0] = 7;
    m2.data[0][1] = 8;
    m2.data[1][0] = 9;
    m2.data[1][1] = 10;
    m2.data[2][0] = 11;
    m2.data[2][1] = 12;
    /*
    [[7,8],
     [9,10],
     [11,12]]
    */
    m2.print();
    Matrix<double> m3 = m1 * m2;
    /*
    [[58, 64],
     [139, 154]]
    */
    m3.print();
    Matrix<double> m4 = m1.transpose();
    /*
    [[1,4],
     [2,5],
     [3,6]]
    */
    m4.print();
    Matrix<double> m5 = m1 * 2;
    /*
    [[2,4,6],
     [8,10,12]]
    */
    m5.print();
    return 0;
}