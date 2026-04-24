#include "methods.h"

#include <cmath>
#include <algorithm>

static const double EPS = 1e-12;


// Метод Гаусса без выбора
bool gaussianNoPivot(Matrix A, Vector b, Vector& x) {
    int n = static_cast<int>(A.size());
    x.assign(n, 0.0);

    // Прямой ход:
    // зануляем элементы под диагональю
    for (int k = 0; k < n; ++k) {
        // Если диагональный элемент слишком мал,
        // метод не может надёжно продолжать
        if (std::abs(A[k][k]) < EPS) {
            return false;
        }

        for (int i = k + 1; i < n; ++i) {
            double factor = A[i][k] / A[k][k];

            for (int j = k; j < n; ++j) {
                A[i][j] -= factor * A[k][j];
            }

            b[i] -= factor * b[k];
        }
    }

    // Обратный ход:
    // решаем верхнетреугольную систему
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];

        for (int j = i + 1; j < n; ++j) {
            sum -= A[i][j] * x[j];
        }

        if (std::abs(A[i][i]) < EPS) {
            return false;
        }

        x[i] = sum / A[i][i];
    }

    return true;
}


// Метод Гаусса с частичным выбором главного элемента

bool gaussianPartialPivot(Matrix A, Vector b, Vector& x) {
    int n = static_cast<int>(A.size());
    x.assign(n, 0.0);

    for (int k = 0; k < n; ++k) {
        // Ищем строку с максимальным по модулю элементом
        // в текущем столбце k
        int pivotRow = k;
        double maxValue = std::abs(A[k][k]);

        for (int i = k + 1; i < n; ++i) {
            if (std::abs(A[i][k]) > maxValue) {
                maxValue = std::abs(A[i][k]);
                pivotRow = i;
            }
        }

        // Если опорный элемент слишком мал ошибка
        if (maxValue < EPS) {
            return false;
        }

        // Меняем строки местами
        if (pivotRow != k) {
            std::swap(A[k], A[pivotRow]);
            std::swap(b[k], b[pivotRow]);
        }

        // Зануляем элементы ниже диагонали
        for (int i = k + 1; i < n; ++i) {
            double factor = A[i][k] / A[k][k];

            for (int j = k; j < n; ++j) {
                A[i][j] -= factor * A[k][j];
            }

            b[i] -= factor * b[k];
        }
    }

    // Обратная подстановка
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];

        for (int j = i + 1; j < n; ++j) {
            sum -= A[i][j] * x[j];
        }

        if (std::abs(A[i][i]) < EPS) {
            return false;
        }

        x[i] = sum / A[i][i];
    }

    return true;
}


// LU-разложение без перестановок
bool luDecomposition(const Matrix& A, Matrix& L, Matrix& U) {
    int n = static_cast<int>(A.size());

    L.assign(n, std::vector<double>(n, 0.0));
    U.assign(n, std::vector<double>(n, 0.0));

    // На диагонали L стоят единицы
    for (int i = 0; i < n; ++i) {
        L[i][i] = 1.0;
    }

    // Алгоритм Дулиттла:
    // строим U по строкам, L по столбцам
    for (int i = 0; i < n; ++i) {
        // Вычисляем i-ю строку матрицы U
        for (int j = i; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < i; ++k) {
                sum += L[i][k] * U[k][j];
            }
            U[i][j] = A[i][j] - sum;
        }

        // Проверка на нулевой диагональный элемент U
        if (std::abs(U[i][i]) < EPS) {
            return false;
        }

        // Вычисляем i-й столбец матрицы L
        for (int j = i + 1; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < i; ++k) {
                sum += L[j][k] * U[k][i];
            }
            L[j][i] = (A[j][i] - sum) / U[i][i];
        }
    }

    return true;
}


// Прямая подстановка
bool forwardSubstitution(const Matrix& L, const Vector& b, Vector& y) {
    int n = static_cast<int>(L.size());
    y.assign(n, 0.0);

    for (int i = 0; i < n; ++i) {
        double sum = b[i];

        for (int j = 0; j < i; ++j) {
            sum -= L[i][j] * y[j];
        }

        if (std::abs(L[i][i]) < EPS) {
            return false;
        }

        y[i] = sum / L[i][i];
    }

    return true;
}


// Обратная подстановка
bool backSubstitution(const Matrix& U, const Vector& y, Vector& x) {
    int n = static_cast<int>(U.size());
    x.assign(n, 0.0);

    for (int i = n - 1; i >= 0; --i) {
        double sum = y[i];

        for (int j = i + 1; j < n; ++j) {
            sum -= U[i][j] * x[j];
        }

        if (std::abs(U[i][i]) < EPS) {
            return false;
        }

        x[i] = sum / U[i][i];
    }

    return true;
}


// Решение через LU целиком
bool solveWithLU(const Matrix& A, const Vector& b, Vector& x) {
    Matrix L, U;

    if (!luDecomposition(A, L, U)) {
        return false;
    }

    Vector y;
    if (!forwardSubstitution(L, b, y)) {
        return false;
    }

    if (!backSubstitution(U, y, x)) {
        return false;
    }

    return true;
}


// Решение через уже вычисленные L и U
bool solveWithReadyLU(const Matrix& L, const Matrix& U, const Vector& b, Vector& x) {
    Vector y;

    if (!forwardSubstitution(L, b, y)) {
        return false;
    }

    if (!backSubstitution(U, y, x)) {
        return false;
    }

    return true;
}
