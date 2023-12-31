#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>

using namespace std;

#define SIGMOID(x) 1 / (1 + exp(-x))
#define SIGMOID_DERIVATIVE(x) exp(x) / pow((exp(x) + 1), 2)
#define Print(x) cout << x << endl

void PrintError(const string error) {
	cout << "\033[31m[Error]: " << "\033[0m" << error << endl;
}

double GetRandomDouble() {
	return (double)(rand() % 20000) / 10000.0 - 1;
}

struct Neuron {
	double value;
	vector<double> weights;

	Neuron(double value) { this->value = value; }
};

class NeuralNetwork {
public:
	NeuralNetwork() {}

	NeuralNetwork(vector<unsigned int> layerInfo) {
		for (int i = 0; i < layerInfo.size(); i++) {
			int x = layerInfo[i];
			//Generate Biases
			this->biases.emplace_back((i == 0) ? 0.0 : GetRandomDouble());
			this->neurons.emplace_back(vector<Neuron>());
			for (int j = 0; j < x; j++) {
				//Generate Neurons
				neurons[neurons.size() - 1].emplace_back(Neuron(GetRandomDouble()));
				if (i == layerInfo.size() - 1) continue;

				for (int k = 0; k < layerInfo[i + 1]; k++) //HAHAHAHAHAHAAHAH
					neurons.back().back().weights.emplace_back(GetRandomDouble());
			}
		}
	}

	void ForwardPropagate() {
		for (int i = 1; i < neurons.size(); i++) {
			auto& currentLayer = neurons[i];
			auto& lastLayer = neurons[i - 1];

			double total = biases[i];
			for (int j = 0; j < currentLayer.size(); j++) {
				for (int k = 0; k < lastLayer.size(); k++)
					total += lastLayer[k].weights[j] * lastLayer[k].value;

				//do the funny sigmoid or whatever
				total = SIGMOID(total);
				currentLayer[j].value = total;
			}
		}
	}

	void SetInput(vector<double> inputs) {
		for (int i = 0; i < inputs.size(); i++)
			neurons[0][i].value = inputs[i];
	}

	void SetOutput() {

	}

	//Returns: Nothing | Sets the values of the input and output for usage in the backpropagation
	void SetExpectedValues(vector<double> inputs, vector<double> outputs) {
		try {
			if (neurons.front().size() != inputs.size() || neurons.back().size() != outputs.size())
				throw exception("The amount of inputs or outputs doesn't match the neural network's schema");

			for (int i = 0; i < inputs.size(); i++)
				neurons[0][i].value = inputs[i];

			expectedOutputs = outputs;

		}
		catch (exception& e) {
			PrintError(e.what());
		}
		catch (...) {
			PrintError("Unexpected error occurred");
		}
	}

	//Returns: nothing | Backpropagates using the expected outputs with a given learningRate
	void Backpropagate(double learningRate) {
		//First get the output error values
		vector<double> outputError;
		for (int i = 0; i < neurons.back().size(); i++) {
			//Formula: (output - target) * derivative(output)
			outputError.emplace_back((neurons.back()[i].value - expectedOutputs[i]) * SIGMOID_DERIVATIVE(neurons.back()[i].value));
		}

		//Calculate error for each layer, we're working our way BACKwards
		vector<vector<double>> errors;
		//Fill out the error vector with an empty vector per layer, afterwards add the output error vector
		for (int i = 0; i < neurons.size() - 1; i++)
			errors.emplace_back(vector<double>());
		errors.push_back(outputError); 

		for (int i = neurons.size() - 2; i > 0; i--) { //layer
			for (int j = 0; j < neurons[i].size(); j++) { //neuron
				double derivative = SIGMOID_DERIVATIVE(neurons[i][j].value);
				double error = 0.0;

				for (int k = 0; k < neurons[i][j].weights.size(); k++) { // weights 
					error += derivative * neurons[i][j].weights[k] * errors[i + 1][k];
				} 

				errors[i].emplace_back(error);
			}
		}
			
		//Update the weights
		for (int i = 0; i < neurons.size() - 1; i++) {
			for (int j = 0; j < neurons[i].size(); j++) {
				for (int k = 0; k < neurons[i][j].weights.size(); k++) {
					neurons[i][j].weights[k] -= learningRate * neurons[i][j].value * errors[i + 1][k];
				}
			}
		}

		//Update the biases
		for (int i = 1; i < neurons.size(); i++) {
			double error = 0.0;
			for (int j = 0; j < errors[i].size(); j++) {
				error += errors[i][j];
			}

			biases[i] -= learningRate * error;
		}
	}

	void GetWeights() {

	}

	void ToString() {
		for (int i = 0; i < neurons.size(); i++) {
			Print("Layer Number: " + to_string(i));
			Print("Bias: " + to_string(biases[i]));

			for (int j = 0; j < neurons[i].size(); j++) {
				Print("    Neuron " +  to_string(j) + " Value: " + to_string(neurons[i][j].value));
				for (int k = 0; k < neurons[i][j].weights.size(); k++) {
					Print("        Weight " + to_string(k) + " Value: " + to_string(neurons[i][j].weights[k]));
				}
			}
		}
	}

	void PrintOutput() {
		for (int i = 0; i < neurons.back().size(); i++)
			cout << "Neuron " << to_string(i) << ": " << neurons.back()[i].value << endl;
	}

	string ToString(int layer) {

	}

private:
	vector<vector<Neuron>> neurons; vector<double> biases, expectedOutputs;
};

vector<string> SplitString(const string input, const char delimiter) {
	vector<std::string> ret;

	for (char c : input) {
		if (c == delimiter) {
			ret.emplace_back("");
			continue;
		}
		if (ret.size() == 0) ret.emplace_back("");
		ret[ret.size() - 1] += c;
	}

	return ret;
}


pair<vector<vector<double>>, vector<vector<double>>> GetTrainingData(string fileName) {
	vector<vector<double>> output; vector<vector<double>> input;

	std::ifstream file(fileName);

	if (file.is_open()) {
		std::string line;
		while (std::getline(file, line)) {
			auto split = SplitString(line, ',');
			output.emplace_back(vector<double>());
			for (int i = 0; i < 8; i++) {
				if (i == stoi(split[0]))
					output.back().emplace_back(1.0);
				else
					output.back().emplace_back(0.0);
			}
			input.emplace_back(vector<double>());

			for (int i = 1; i < split.size(); i++) {
				input.back().emplace_back(stol(split[i]));
			}
		}
		file.close();
	}
	else {
		std::cerr << "Unable to open file" << std::endl;
	}

	return make_pair(input, output);
}

int main() {
	srand(time(NULL));
	vector<unsigned int> layerInfo = { 3, 24, 16, 12, 8 }; 
	NeuralNetwork network(layerInfo);

	auto trainData = GetTrainingData("lol.csv");

	for (int i = 0; i < 1000; i++) {
		auto inputs = trainData.first;
		auto outputs = trainData.second;

		for (int j = 0; j < outputs.size(); j++) {
			network.SetExpectedValues(inputs[j], outputs[j]);
			network.ForwardPropagate();
			network.Backpropagate(0.05);
		}
	}

	auto testData = GetTrainingData("lol.csv");

	network.SetInput(vector<double> { 0, 0, 1 });
	network.ForwardPropagate();
	network.PrintOutput();
	cout << "--------------------------------------" << endl;

	network.SetInput(vector<double> { 1, 0, 0 });
	network.ForwardPropagate();
	network.PrintOutput();
	cout << "--------------------------------------" << endl;

	network.SetInput(vector<double> { 1, 1, 1 });
	network.ForwardPropagate();
	network.PrintOutput();
	cout << "--------------------------------------" << endl;
 
	
	system("pause");
	return 0;
}