#include "experiments.h"

#include "matrix.h"
#include "methods.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;
using Microseconds = std::chrono::microseconds;


// Эксперимент 1: одна система
void runSingleSystemExperiment() {
    std::cout << "Эксперимент 1: одна система\n";

    std::ofstream csv("results.csv");
    csv << "experiment,n,method,time_us,residual\n";

    // Фиксированный seed для воспроизводимости
    std::mt19937 gen(42);

    int sizes[] = {100, 200, 500, 1000};

    std::cout << std::left
              << std::setw(8)  << "n"
              << std::setw(20) << "Method"
              << std::setw(18) << "Time (us)"
              << std::setw(18) << "Residual"
              << "\n";

    for (int n : sizes) {
        Matrix A = generateRandomMatrix(n, gen);
        Vector b = generateRandomVector(n, gen);
        Vector x;

        // Гаусс без выбора
        {
            auto start = Clock::now();
            bool ok = gaussianNoPivot(A, b, x);
            auto end = Clock::now();

            long long time_us =
                std::chrono::duration_cast<Microseconds>(end - start).count();

            double resid = ok ? residualNorm(A, x, b) : -1.0;

            std::cout << std::setw(8)  << n
                      << std::setw(20) << "Gauss no pivot"
                      << std::setw(18) << time_us
                      << std::setw(18) << resid
                      << "\n";

            csv << "single," << n << ",Gauss_no_pivot," << time_us << "," << resid << "\n";
        }

        // Гаусс с выбором
        {
            auto start = Clock::now();
            bool ok = gaussianPartialPivot(A, b, x);
            auto end = Clock::now();

            long long time_us =
                std::chrono::duration_cast<Microseconds>(end - start).count();

            double resid = ok ? residualNorm(A, x, b) : -1.0;

            std::cout << std::setw(8)  << n
                      << std::setw(20) << "Gauss pivot"
                      << std::setw(18) << time_us
                      << std::setw(18) << resid
                      << "\n";

            csv << "single," << n << ",Gauss_pivot," << time_us << "," << resid << "\n";
        }

        // LU: отдельно разложение и отдельно решение
        {
            Matrix L, U;

            auto startDecomp = Clock::now();
            bool okDecomp = luDecomposition(A, L, U);
            auto endDecomp = Clock::now();

            long long decomp_us =
                std::chrono::duration_cast<Microseconds>(endDecomp - startDecomp).count();

            long long solve_us = -1;
            double resid = -1.0;

            if (okDecomp) {
                auto startSolve = Clock::now();
                bool okSolve = solveWithReadyLU(L, U, b, x);
                auto endSolve = Clock::now();

                solve_us =
                    std::chrono::duration_cast<Microseconds>(endSolve - startSolve).count();

                if (okSolve) {
                    resid = residualNorm(A, x, b);
                }
            }

            std::cout << std::setw(8)  << n
                      << std::setw(20) << "LU decomp"
                      << std::setw(18) << decomp_us
                      << std::setw(18) << "-"
                      << "\n";

            std::cout << std::setw(8)  << n
                      << std::setw(20) << "LU solve"
                      << std::setw(18) << solve_us
                      << std::setw(18) << resid
                      << "\n";

            csv << "single," << n << ",LU_decomp," << decomp_us << "," << -1 << "\n";
            csv << "single," << n << ",LU_solve,"  << solve_us  << "," << resid << "\n";
            csv << "single," << n << ",LU_total,"  << (decomp_us + solve_us) << "," << resid << "\n";
        }

        std::cout << "\n";
    }

    csv.close();
}


// Эксперимент 2: несколько правых частей
void runMultipleRHSExperiment() {
    std::cout << "Эксперимент 2: несколько правых частей\n";
  
    std::ofstream csv("results.csv", std::ios::app);

    std::mt19937 gen(42);

    int n = 500;
    int ks[] = {1, 10, 100};

    Matrix A = generateRandomMatrix(n, gen);

    std::cout << std::left
              << std::setw(8)  << "k"
              << std::setw(22) << "Method"
              << std::setw(18) << "Total time (us)"
              << "\n";

    // Для каждого k генерируем k правых частей
    for (int k : ks) {
        std::vector<Vector> rhsList;
        for (int i = 0; i < k; ++i) {
            rhsList.push_back(generateRandomVector(n, gen));
        }

        // Гаусс с выбором — каждый раз решаем заново
        {
            auto start = Clock::now();

            for (int i = 0; i < k; ++i) {
                Vector x;
                gaussianPartialPivot(A, rhsList[i], x);
            }

            auto end = Clock::now();

            long long total_us =
                std::chrono::duration_cast<Microseconds>(end - start).count();

            std::cout << std::setw(8)  << k
                      << std::setw(22) << "Gauss pivot"
                      << std::setw(18) << total_us
                      << "\n";

            csv << "multi_rhs," << k << ",Gauss_pivot_total," << total_us << "," << -1 << "\n";
        }

        // LU — один раз разложение, потом много подстановок
        {
            Matrix L, U;

            auto start = Clock::now();

            bool ok = luDecomposition(A, L, U);
            if (ok) {
                for (int i = 0; i < k; ++i) {
                    Vector x;
                    solveWithReadyLU(L, U, rhsList[i], x);
                }
            }

            auto end = Clock::now();

            long long total_us =
                std::chrono::duration_cast<Microseconds>(end - start).count();

            std::cout << std::setw(8)  << k
                      << std::setw(22) << "LU total"
                      << std::setw(18) << total_us
                      << "\n";

            csv << "multi_rhs," << k << ",LU_total," << total_us << "," << -1 << "\n";
        }

        std::cout << "\n";
    }

    csv.close();
}


// Эксперимент 3: матрица Гильберта
void runHilbertExperiment() {
    std::cout << "Эксперимент 3: точность на матрице Гильберта\n";

    std::ofstream csv("results.csv", std::ios::app);

    int sizes[] = {5, 10, 15};

    std::cout << std::left
              << std::setw(8)  << "n"
              << std::setw(20) << "Method"
              << std::setw(20) << "Rel. error"
              << std::setw(20) << "Residual"
              << "\n";

    for (int n : sizes) {
        Matrix H = generateHilbertMatrix(n);

        // Точное решение x = (1,1,...,1)^T
        Vector x_exact(n, 1.0);

        // Правая часть b = H * x_exact
        Vector b = multiplyMatrixVector(H, x_exact);

        Vector x;

        // Гаусс без выбора
        {
            bool ok = gaussianNoPivot(H, b, x);

            double err = ok ? relativeError(x, x_exact) : -1.0;
            double resid = ok ? residualNorm(H, x, b) : -1.0;

            std::cout << std::setw(8)  << n
                      << std::setw(20) << "Gauss no pivot"
                      << std::setw(20) << err
                      << std::setw(20) << resid
                      << "\n";

            csv << "hilbert," << n << ",Gauss_no_pivot," << err << "," << resid << "\n";
        }

        // Гаусс с выбором
        {
            bool ok = gaussianPartialPivot(H, b, x);

            double err = ok ? relativeError(x, x_exact) : -1.0;
            double resid = ok ? residualNorm(H, x, b) : -1.0;

            std::cout << std::setw(8)  << n
                      << std::setw(20) << "Gauss pivot"
                      << std::setw(20) << err
                      << std::setw(20) << resid
                      << "\n";

            csv << "hilbert," << n << ",Gauss_pivot," << err << "," << resid << "\n";
        }

        // LU без перестановок
        {
            bool ok = solveWithLU(H, b, x);

            double err = ok ? relativeError(x, x_exact) : -1.0;
            double resid = ok ? residualNorm(H, x, b) : -1.0;

            std::cout << std::setw(8)  << n
                      << std::setw(20) << "LU"
                      << std::setw(20) << err
                      << std::setw(20) << resid
                      << "\n";

            csv << "hilbert," << n << ",LU," << err << "," << resid << "\n";
        }

        std::cout << "\n";
    }

    csv.close();
}
