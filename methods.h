#pragma once

#include "matrix.h"

// Метод Гаусса без выбора главного элемента
// Возвращает true, если решение найдено успешно
bool gaussianNoPivot(Matrix A, Vector b, Vector& x);

// Метод Гаусса с частичным выбором главного элемента по столбцу
bool gaussianPartialPivot(Matrix A, Vector b, Vector& x);

// LU-разложение без перестановок
// A = L * U
// Возвращает true, если разложение удалось
bool luDecomposition(const Matrix& A, Matrix& L, Matrix& U);

// Прямая подстановка для решения L*y = b
bool forwardSubstitution(const Matrix& L, const Vector& b, Vector& y);

// Обратная подстановка для решения U*x = y
bool backSubstitution(const Matrix& U, const Vector& y, Vector& x);

// Решение через LU: сначала разложение, потом 2 подстановки
bool solveWithLU(const Matrix& A, const Vector& b, Vector& x);

// Решение через уже готовые L и U
// Для эксперимента с несколькими правыми частями
bool solveWithReadyLU(const Matrix& L, const Matrix& U, const Vector& b, Vector& x);


