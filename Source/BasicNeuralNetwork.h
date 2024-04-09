#pragma once

#ifndef NEURAL_NET
#define NEURAL_NET

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <string>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::to_string;

typedef vector<vector<double>> DoubleMatrix;

namespace bsn {
	class NeuralNetwork;

	namespace functional {
		class ActivationFunction {
		public:
			virtual double Function(double) const = 0;
			virtual double Derivative(double) const = 0;
		};

		class Sigmoid : public ActivationFunction {
		public:
			Sigmoid() = default;

			double Function(double x) const {
				return (1.0 / (1.0 + exp(-(x))));
			}

			double Derivative(double x) const {
				return (Function(x) * (1 - Function(x)));
			}
		};

		class ReLU : public ActivationFunction {
		public:
			ReLU() = default;

			double Function(double x) const {
				return ((x) > 0 ? (x) : 0.0);
			}

			double Derivative(double x) const {
				return ((x) > 0 ? 1.0 : 0.0);
			}
		};

		class TanH : public ActivationFunction {
		public:
			TanH() = default;

			double Function(double x) const {
				return tanh(x);
			}

			double Derivative(double x) const {
				double th = tanh(x);
				return 1.0 - th * th;
			}
		};

		double HeInitialization(unsigned int prev_layer_size) {
			static std::random_device rd;
			static std::mt19937 eng(rd());
			double stddev = std::sqrt(2.0 / prev_layer_size);
			std::normal_distribution<> distr(0, stddev);

			return distr(eng);
		}
	};

	namespace helpers {
		vector<string> SplitString(const string input, const char delimiter) {
			vector<string> ret;

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

		int GetHighestIndex(vector<double> vec) {
			int indexMax = -1; double max = -100000.0;
			for (int i = 0; i < vec.size(); i++)
				if (vec[i] > max) {
					max = vec[i]; indexMax = i;
				}

			return indexMax;
		}
	};

	struct Neuron {
		Neuron() = default;
		Neuron(double value) { this->value = value; }
		vector<double> weights;
		double value = 0;
	};

	class NeuralNetwork {
	public:
		NeuralNetwork() = default;

		NeuralNetwork(const std::vector<unsigned int>& layerInfo) {
			for (size_t i = 0; i < layerInfo.size(); ++i) {
				biases.emplace_back(std::vector<double>(layerInfo[i], (i == 0) ? 0.0 : 1.0));
				neurons.emplace_back(std::vector<Neuron>(0.0));
				errors.emplace_back(std::vector<double>(layerInfo[i], 0.0));

				for (size_t j = 0; j < layerInfo[i]; ++j) {
					neurons.back().emplace_back(Neuron(0.0));

					if (i < layerInfo.size() - 1)
						for (size_t k = 0; k < layerInfo[i + 1]; ++k)
							neurons.back().back().weights.emplace_back(functional::HeInitialization(layerInfo[i]));
				}
			}
		}

		//Delete the activation function on destruction
		~NeuralNetwork() {
			delete f;
		}

		//Returns: Nothing | Forward passes the values
		void ForwardPropagate() {
			for (int i = 1; i < neurons.size(); i++) {
				for (int j = 0; j < neurons[i].size(); j++) {
					double total = biases[i][j];
					for (int k = 0; k < neurons[i - 1].size(); k++)
						total += neurons[i - 1][k].weights[j] * neurons[i - 1][k].value;

					//Use the activation function on the neuron's value
					neurons[i][j].value = f->Function(total);
				}
			}
		}

		//Returs: Nothing | Define what activation function should be used
		void SetActivationFunction(functional::ActivationFunction* a) {
			delete f; f = a;
		}

		//Returns: Nothing | Set the values of the input layer
		void SetInput(vector<double> inputs) {
			for (int i = 0; i < inputs.size(); i++)
				neurons[0][i].value = inputs[i];
		}

		//Returns: Nothing | Sets the values of the input and output for usage in the backpropagation
		void SetExpectedValues(vector<double> inputs, vector<double> outputs) {
			if (neurons.front().size() != inputs.size() || neurons.back().size() != outputs.size())
				return;

			for (int i = 0; i < inputs.size(); i++)
				neurons[0][i].value = inputs[i];

			expectedOutputs = outputs;
		}

		//Returns: The error values given the inputs
		DoubleMatrix GetErrors() const {
			return errors;
		}

		//Returns: Nothing
		void SetErrors(DoubleMatrix errors) {
			this->errors = errors;
		}

		//Returns: error matrix | Calculates the errors based on the expected outputs
		DoubleMatrix CalculateError() {
			//First get the output error values
			for (int i = 0; i < neurons.back().size(); i++) {
				//Formula: (output - target) * derivative(output)
				const double value = neurons.back()[i].value;
				errors.back()[i] = (value - expectedOutputs[i]) * f->Derivative(value);
			}

			//Calculate error for each layer, we're working our way BACKwards
			for (int i = neurons.size() - 2; i > 0; i--) { //layer
				for (int j = 0; j < neurons[i].size(); j++) { //neuron
					const double derivative = f->Derivative(neurons[i][j].value);
					double error = 0.0;

					for (int k = 0; k < neurons[i][j].weights.size(); k++) { // weights 
						error += derivative * neurons[i][j].weights[k] * errors[i + 1][k];
					}

					errors[i][j] = error;
				}
			}

			return errors;
		}

		//Returns: Nothing | Backpropagates using the error values with a given learningRate
		void Backpropagate(const double learningRate) {
			//Update the weights
			for (int i = 0; i < neurons.size() - 1; i++) {
				for (int j = 0; j < neurons[i].size(); j++) {
					for (int k = 0; k < neurons[i][j].weights.size(); k++) {
						neurons[i][j].weights[k] -= learningRate * (neurons[i][j].value * errors[i + 1][k] + 0.001 * neurons[i][j].weights[k]);
					}
				}
			}

			//Update the biases
			for (int i = 1; i < neurons.size(); i++)
				for (int j = 0; j < errors[i].size(); j++)
					biases[i][j] -= learningRate * errors[i][j];
		}

		void LoadWeightsFromFile(const string fileName) {
			vector<string> ret; string text; std::ifstream file(fileName);

			while (getline(file, text))
				ret.emplace_back(text);

			file.close();
			for (int i = 0; i < ret.size(); i++) { //layer
				auto n = helpers::SplitString(ret[i], '|');
				for (int j = 0; j < n.size(); j++) { //neurons
					auto s = helpers::SplitString(n[j], ' ');
					biases[i][j] = std::stod(s[0]); //bias

					for (int k = 0; k < neurons[i][j].weights.size(); k++) // weights
						neurons[i][j].weights[k] = std::stod(s[k + 1]);
				}
			}
		}

		void SaveWeightsToFile(const string fileName) {
			std::ofstream file(fileName, std::ios::trunc);
			for (int i = 0; i < neurons.size(); i++) {
				string data = "";
				for (int j = 0; j < neurons[i].size(); j++) {
					data += to_string(biases[i][j]) + " ";
					for (int k = 0; k < neurons[i][j].weights.size(); k++) {
						if (k == neurons[i][j].weights.size() - 1)
							data += to_string(neurons[i][j].weights[k]);
						else
							data += to_string(neurons[i][j].weights[k]) + " ";
					}
					if (j != neurons[i].size() - 1)
						data += "|";
				}
				file << data << endl;
			}
			file.close();
		}

		//Returns the last layer / output neurons 
		vector<double> GetOutputs() const {
			vector<double> ret;
			for (const auto x : neurons.back())
				ret.emplace_back(x.value);
			return ret;
		}

		//Returns: The neural network in a formatted string
		string ToString() const {
			std::stringstream ss;
			for (int i = 0; i < neurons.size(); i++) {
				ss << "Layer Number: " << i << "\n";

				for (int j = 0; j < neurons[i].size(); j++) {
					ss << "    Bias: " << biases[i][j] << "\n";
					ss << "    Neuron " << j << " Value: " << neurons[i][j].value << "\n";
					for (int k = 0; k < neurons[i][j].weights.size(); k++) {
						ss << "        Weight " << k << " Value: " << neurons[i][j].weights[k] << "\n";
					}
				}
			}
			return ss.str();
		}

	private:
		vector<vector<Neuron>> neurons; vector<double> expectedOutputs;
		DoubleMatrix biases, errors;
		functional::ActivationFunction* f = new functional::ReLU();
	};

	void TrainNetwork(NeuralNetwork& network, const DoubleMatrix inputs, const DoubleMatrix outputs, const int epochs, const double learningRate, const double decayRate) {
		double rate = learningRate;

		for (size_t i = 0; i < epochs; i++) {
			for (size_t j = 0; j < outputs.size(); j++) {
				network.SetExpectedValues(inputs[j], outputs[j]);
				network.ForwardPropagate();
				network.CalculateError();
				network.Backpropagate(rate);
			}

			rate *= decayRate;
		}
	}

	double GetSuccessRate(NeuralNetwork& network, const DoubleMatrix inputs, const DoubleMatrix outputs) {
		unsigned int unsuccessful = 0;
		for (int i = 0; i < inputs.size(); i++) {
			cout << "-------------------------------------- Expected: " << helpers::GetHighestIndex(outputs[i]) << endl;
			network.SetInput(inputs[i]);
			network.ForwardPropagate();
			cout << "Actual: " << helpers::GetHighestIndex(network.GetOutputs()) << endl;

			if (helpers::GetHighestIndex(network.GetOutputs()) != helpers::GetHighestIndex(outputs[i]))
				unsuccessful++;
		}

		return (1.0 - (double)unsuccessful / outputs.size());
	}
};

#endif // !NEURAL_NET

