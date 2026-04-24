#pragma once

#include <vector>
#include <random>


// Matrix = двумерный вектор (матрица)
// Vector = одномерный вектор
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// Генерация случайной матрицы n x n
Matrix generateRandomMatrix(int n, std::mt19937& gen);

// Генерация случайного вектора длины n
Vector generateRandomVector(int n, std::mt19937& gen);

// Построение матрицы Гильберта размера n x n
Matrix generateHilbertMatrix(int n);

// Умножение матрицы на вектор: b = A * x
Vector multiplyMatrixVector(const Matrix& A, const Vector& x);

// Евклидова норма вектора ||v||
double vectorNorm(const Vector& v);

// Невязка ||Ax - b||
double residualNorm(const Matrix& A, const Vector& x, const Vector& b);

// Относительная погрешность ||x_approx - x_exact|| / ||x_exact||
double relativeError(const Vector& x_approx, const Vector& x_exact);


