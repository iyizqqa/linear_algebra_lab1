#include "experiments.h"

#include <iostream>

int main() {
    std::cout << "Лабораторная работа: сравнение методов решения СЛАУ\n\n";

    runSingleSystemExperiment();
    runMultipleRHSExperiment();
    runHilbertExperiment();

    std::cout << "Результаты сохранены в файл results.csv\n";

    return 0;
}
