#include "constants.h"
#include "rand.h"
#include "matrice.h"
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>

// implement classes for neural net computation
template<class T>
void print_vector(const std::vector<T>& vec) {
	for (const auto& elem : vec) {
		std::cout << elem << "\t";
	}
	std::cout << '\n';
}

namespace nn {
	std::vector<double> sigmoid(const std::vector<double>& vals) { 
		std::vector<double> res(vals.size(), 0.0);
		for (int i = 0; i != vals.size(); ++i) {
			res[i] = 1 / (1 + std::pow(constants::euler, vals[i]));
		}
		return res;
	}

	std::vector<double> cross_entropy(const std::vector<int>& labels,const std::vector<double>& activs) {
		assert((labels.size() == activs.size()) && "the length of labels and activs should be same");
		std::vector<double> loss(labels.size(), 0.0);
		for (int i = 0; i != loss.size(); ++i) {
			assert(0.0 <= activs[i] && activs[i] <= 1.0 && "activations should be between 0 and 1");
			loss[i] = -1.0 * ((labels[i])* std::log(labels[i]) + (1.0 - labels[i]) * std::log(1.0 - labels[i]));
		}
		return loss;
	}

	std::vector<double> schur(const std::vector<double>& lhs, const std::vector<double>& rhs) {
		assert((lhs.size() == rhs.size()) && "lengths of the lhs and rhs should be equal");
		std::vector<double> res(lhs.size(), 0.0);
		for (int i = 0; i != lhs.size(); ++i) {
			res[i] = lhs[i] * rhs[i];
		}
		return res;
	}

	class Network {
	private:
		std::vector<int> layers;
		std::vector<std::vector<double>> biases;
		std::vector<matrice::Matrix<double>> weights;
		std::vector<std::vector<double>> activations;
		std::vector<std::vector<double>> zs;	// weighted inputs
	public:
		Network(std::vector<int> layer_dims) : layers{layer_dims} {
			double sigma = 1.0 / layer_dims[0];
			for (int i = 1; i != layer_dims.size(); ++i) {
				biases.push_back(std::vector<double>(layer_dims[i], 0.0));
				activations.push_back(std::vector<double>(layer_dims[i], 0.0));
				weights.push_back(rand_lib::rand_matrix(layer_dims[i], layer_dims[i - 1], 0.0, sigma));
				zs.push_back(std::vector<double>(layer_dims[i], 0.0));
			}
		}
	};

}



int main()
{
	std::vector<int> layers{ 5, 2, 1 };
	nn::Network network(layers);

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
