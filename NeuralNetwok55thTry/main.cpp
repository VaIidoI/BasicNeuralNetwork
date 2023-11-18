#include <iostream>
#include <vector>
#include <random>
#include <string>

using namespace std;

#define SIGMOID(x) 1 / (1 + exp(-x))
#define SIGMOID_DERIVATIVE(x) exp(x) / pow((exp(x) + 1), 2)
#define Print(x) cout << x << endl

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
			this->biases.emplace_back(0.0);
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

	void SetInput() {

	}

	void SetOutput() {

	}

	void Backpropagate() {

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

	string ToString(int layer) {

	}

private:
	vector<vector<Neuron>> neurons; vector<double> biases;
};

int main() {
	srand(time(NULL));
	vector<unsigned int> layerInfo = { 2, 2, 1 }; 
	NeuralNetwork network(layerInfo);
	network.ToString();
	network.ForwardPropagate();
	network.ToString();
	return 0;
}