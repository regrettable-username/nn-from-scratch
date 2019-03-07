#include <array>
#include <fstream>
#include <iostream>
#include <cmath>
#include <tuple>

static const size_t batchSize = 5;
using Tuple = std::tuple<float, float>;
using TrainBatch = std::array<Tuple, batchSize>;

void printTuple(Tuple tuple) {
    auto [x, y] = tuple;
    std::cout << "x: " << x << " y: " << y << std::endl;
}

static const TrainBatch trainingData = { Tuple(1.0, 1.0), Tuple(2.2, 3.0), Tuple(3.3, 3.9), Tuple(4.0, 3.9), Tuple(5.0, 6.0) };

Tuple fitLine(const TrainBatch& x, float w, float b, float lr)
{
    // Used to compute the mean. Includes the derivative of the squred term.
    float batchScale = 2.0 / static_cast<float>(x.size());
    
    // Used to store the gradients.
    float weightGrad = 0;
    float biasGrad = 0;

    // Track the error of each pass.
    float totalSquaredError = 0;
    
    // Start adding the contributions to the gradients.
    for (auto const &value : trainingData) {
        auto [x, y] = value;

        // Evaluate the model and compute the error.
        float y_hat = w * x + b;
        float error = y_hat - y;

        // Add up the errors for tracking.
        totalSquaredError += error * error;
        
        // Add up the grads
        weightGrad += x * error;
        biasGrad += error;
    }
    
    weightGrad *= batchScale;
    biasGrad *= batchScale;

    // Gradient Descent. Making sure to scale lr by batch scale.
    float updatedWeight = w - weightGrad * lr * batchScale;
    float updatedBias = b - biasGrad * lr * batchScale;

    // Log the MSE each time this is called.
    float mse = totalSquaredError * 1.0 / static_cast<float>(x.size());
    std::cout << "MSE: " << mse << std::endl;

    return Tuple(updatedWeight, updatedBias);
}

int main(void) {
    
    // Set some initial values for our linear equation
    float w = 1.5;
    float b = 1.0;
    float lr = 0.1;
    
    // print to stdout
    std::cout << "________Training Data________" << std::endl;

    // Print the training data and write to CSV.
    std::ofstream trainFile;
    trainFile.open("train.csv");
    trainFile << "x" << "," << "y" << std::endl;

    for (auto const &value : trainingData) {
        printTuple(value);
        auto [x, y] = value;
        trainFile << x << ", " << y << "\n";
    }

    trainFile.close();
    
    // Write the original slope and bias to another CSV
    std::ofstream lineFile;
    lineFile.open("line.csv");
    lineFile << "slope" << "," << "bias" << std::endl;
    lineFile << w << "," << b;
    lineFile.close();

    // Fit the line via Gradient Descent.
    Tuple fittedLine = Tuple(w, b);

    for (int i = 0; i < 200; i++) {
        auto [fittedWeight, fittedBias] = fittedLine;
        fittedLine = fitLine(trainingData, fittedWeight, fittedBias, lr);
    }

    // Write the fitted line to a CSV for plotting.
    auto [fittedWeight, fittedBias] = fittedLine;
    std::ofstream fittedLineFile;
    fittedLineFile.open("fitted_line.csv");
    fittedLineFile << "slope" << "," << "bias" << std::endl;
    fittedLineFile << fittedWeight << "," << fittedBias << std::endl;
    fittedLineFile.close();

    return 0;
}
