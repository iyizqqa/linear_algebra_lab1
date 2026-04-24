#include "matrix.h"

#include <cmath>
#include <stdexcept>

Matrix generateRandomMatrix(int n, std::mt19937& gen) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Matrix A(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = dist(gen);
        }
    }

    return A;
}

Vector generateRandomVector(int n, std::mt19937& gen) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Vector b(n);
    for (int i = 0; i < n; ++i) {
        b[i] = dist(gen);
    }

    return b;
}

Matrix generateHilbertMatrix(int n) {
    Matrix H(n, std::vector<double>(n));

    // Формула матрицы Гильберта:
    // H[i][j] = 1 / (i + j + 1), если индексация с нуля
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            H[i][j] = 1.0 / (i + j + 1.0);
        }
    }

    return H;
}

Vector multiplyMatrixVector(const Matrix& A, const Vector& x) {
    int n = static_cast<int>(A.size());

    if (static_cast<int>(x.size()) != n) {
        throw std::runtime_error("Размеры матрицы и вектора не совпадают");
    }

    Vector result(n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i] += A[i][j] * x[j];
        }
    }

    return result;
}

double vectorNorm(const Vector& v) {
    double sum = 0.0;

    for (double value : v) {
        sum += value * value;
    }

    return std::sqrt(sum);
}

double residualNorm(const Matrix& A, const Vector& x, const Vector& b) {
    Vector Ax = multiplyMatrixVector(A, x);

    Vector r(b.size());
    for (size_t i = 0; i < b.size(); ++i) {
        r[i] = Ax[i] - b[i];
    }

    return vectorNorm(r);
}

double relativeError(const Vector& x_approx, const Vector& x_exact) {
    if (x_approx.size() != x_exact.size()) {
        throw std::runtime_error("Размеры векторов не совпадают");
    }

    Vector diff(x_exact.size());
    for (size_t i = 0; i < x_exact.size(); ++i) {
        diff[i] = x_approx[i] - x_exact[i];
    }

    double denom = vectorNorm(x_exact);
    if (denom == 0.0) {
        throw std::runtime_error("Норма точного решения равна нулю");
    }

    return vectorNorm(diff) / denom;
}
