<h1 align="center">Simple CPU-based Neural Network Library</h1>

<p align="center">
A lightweight, header-only C++ neural network library designed for educational purposes and small projects. This library focuses on simplicity and readability, avoiding complex dependencies like CUDA.
</p>

## Features

- **Header-Only**: Easy to include in any project without the need for compiling separate libraries.
- **CPU-Based**: Runs on the CPU, making it accessible for users without high-end GPUs and keeps code complexity down.
- **Simple Activation Functions**: Includes basic activation functions like Sigmoid, ReLU, and TanH to explore different neural behaviors.
- **Customizable Network Architecture**: Allows for easy adjustments of layers and neurons to experiment with different network configurations.

## Getting Started

To integrate this neural network library into your project, simply include the `BasicNeuralNetwork.h` file in your source code. Ensure C++11 or later is used due to the usage of certain language features.

### Example: Creating and Training a Neural Network

Below is a quick start guide on setting up a simple neural network, training it with sample data, and evaluating its performance:

```cpp
#include "BasicNeuralNetwork.h"
#include <iostream>

using namespace bsn;

int main() {
    // Define the structure of the neural network: input layer, hidden layers, output layer
    vector<unsigned int> layerInfo = { inputSize, 16, 8, outputSize }; 
    NeuralNetwork network(layerInfo);

    // Optional: Load pre-trained weights
    // network.LoadWeightsFromFile("weights.txt");
    
    // Set the activation function (ReLU in this case)
    network.SetActivationFunction(new functional::ReLU());

    // Prepare your training and test data
    auto trainData = GetCustomData("training_data.csv");
    auto testData = GetCustomData("test_data.csv");

    // Train the network
    TrainNetwork(network, trainData.first, trainData.second, 250, 0.01, 0.995);

    // Save the trained weights
    network.SaveWeightsToFile("weights.txt");

    // Evaluate the network's performance on the test set
    cout << "Success-Rate: " << to_string(GetSuccessRate(network, testData.first, testData.second)) << endl;
    
    return 0;
}
```

## Customization and Extensions

This library is intentionally kept minimalistic for ease of understanding and modification. You're encouraged to extend it with more complex functionalities like additional activation functions, changing the logic and various other changes.

## Contribute

Contributions are welcome! I am severely incompetent so feel free to fix any obvious issues.

## License

This project is open-source and under the WTFPL license.
