#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include "BasicNeuralNetwork.h"

using namespace std;
using namespace bsn;


pair<DoubleMatrix, DoubleMatrix> GetAppleData(const string fileName) {
	DoubleMatrix output; DoubleMatrix input;

	std::ifstream file(fileName);
	size_t index = 0;
	if (file.is_open()) {
		std::string line;
		while (std::getline(file, line)) {
			if (index++ == 0) continue;
			auto split = helpers::SplitString(line, ',');
			output.emplace_back(vector<double>());
			input.emplace_back(vector<double>());

			for (size_t i = 1; i < split.size() - 1; i++)
				input.back().emplace_back(stod(split[i]));

			if (split.back() == "good") {
				output.back().emplace_back(1.0);
				output.back().emplace_back(0.0);
			}
			else {
				output.back().emplace_back(0.0);
				output.back().emplace_back(1.0);
			}
		}
	}
	else {
		std::cerr << "Unable to open file" << std::endl;
	}

	return make_pair(input, output);
}

int main() {
	vector<unsigned int> layerInfo = { 7, 32, 8, 2 }; 
	NeuralNetwork network(layerInfo);
	network.LoadWeightsFromFile("apple.txt");
	network.SetActivationFunction(new functional::ReLU());
	auto trainData = GetAppleData("apple_quality.csv");

	TrainNetwork(network, trainData.first, trainData.second, 300, 0.01, 0.995);

	auto testData = GetAppleData("apple_test.csv");

	network.SaveWeightsToFile("apple.txt");

	cout << "Success-Rate: " << to_string(GetSuccessRate(network, testData.first, testData.second)) << endl;
	
	system("pause");
	return 0;
}