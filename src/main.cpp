#include <iostream>
#include <fstream>
#include <tuple>
#include <array>

static const size_t batchSize = 5;
using Tuple = std::tuple<float, float>;
using TrainBatch = std::array<Tuple, batchSize>;

void printTuple(Tuple tuple) {
    auto [x, y] = tuple;
    std::cout << "x: " << x << " y: " << y << std::endl;
}

static const TrainBatch trainingData = { Tuple(1.0, 1.0), Tuple(2.0, 2.0), Tuple(3.0, 4.0), Tuple(4.0, 2.0), Tuple(5.0, 6.0) };

int main(void) {
    
    // Set some initial values for our linear equation
    float y = 4.0; // Actual Y
    float x = 1.0;

    float w = 1.5;
    float b = 1.0;

    float yPrime = w * x + b;
    
    // print to stdout
    std::cout << "wx + b = " << yPrime << std::endl;
    std::cout << "________Training Data________" << std::endl;

    // Print the training data and write to CSV.
    std::ofstream trainFile;
    trainFile.open("train.csv");
    trainFile << "x" << "," << "y" << std::endl;

    for (auto const &value: trainingData) {
        printTuple(value);
        auto [x, y] = value;
        trainFile << x << ", " << y << "\n";
    }

    trainFile.close();
    
    // Write the slope and bias to a CSV
    std::ofstream lineFile;
    lineFile.open("line.csv");
    lineFile << "slope" << "," << "bias" << std::endl;
    lineFile << w << ", " << b;
    lineFile.close();

    return 0;
}
