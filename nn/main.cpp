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

template <typename T>
void print_matrix(const matrice::Matrix<T>& mat) {
	auto data = mat.get_data();
	for (int i = 0; i < mat.get_row_num(); ++i) {
		for (int j = 0; j < mat.get_col_num(); ++j) {
			std::cout << data[i][j] << ", ";
		}
		std::cout << "\n";
	}
}

namespace nn {
	template<typename T>
	T sigmoid(T in) {
		T out = 1 / (1 + std::pow(constants::euler, -in));
		return out;
	}
	template <typename T>
	matrice::Matrix<T> sigmoid(const matrice::Matrix<T>& in) {
		matrice::Matrix<double> out(in.get_row_num(), in.get_col_num(), 0.0);
		for (int i = 0; i < out.get_row_num(); ++i) {
			for (int j = 0; j < out.get_col_num(); ++j) {
				T val = sigmoid(in.get_val(i, j));
				out.assign_val(i, j, val);
			}
		}
		return out;
	}

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
		int layer_num;
		std::vector<matrice::Matrix<double>> biases;
		std::vector<matrice::Matrix<double>> weights;
		std::vector<matrice::Matrix<double>> activations;
		std::vector<matrice::Matrix<double>> zs;	// weighted inputs
	public:
		Network(std::vector<int> layer_dims) : layers{ layer_dims }{
			layer_num = layer_dims.size() - 1;
			double sigma = 1.0 / layer_dims[0];
			for (int i = 1; i != layer_dims.size(); ++i) {
				biases.push_back(matrice::Matrix<double>(layer_dims[i],1, 0.0));
				activations.push_back(matrice::Matrix<double>(layer_dims[i], 1, 0.0));
				weights.push_back(rand_lib::rand_matrix(layer_dims[i], layer_dims[i - 1], 0.0, sigma));
				zs.push_back(matrice::Matrix<double>(layer_dims[i], 1, 0.0));
			}
		}
		int get_layer_num() const { return layer_num; }
		matrice::Matrix<double> get_bias(int index) const {
			assert((0 <= index && index < layers.size()) && "index out of range");
			return biases[index];
		}
		matrice::Matrix<double> get_weight(int index) const { 
			assert((0 <= index && index < layers.size()) && "index out of range");
			return weights[index]; 
		}
		matrice::Matrix<double> get_activation(int index) const {
			assert((0 <= index && index < layers.size()) && "index out of range");
			return activations[index];
		}
		matrice::Matrix<double> get_z(int index) const {
			assert((0 <= index && index < layers.size()) && "index out of range");
			return zs[index];
		}

		void assign_z(const matrice::Matrix<double>& z, int index) {
			zs[index] = z;
		}

		void assign_activ(const matrice::Matrix<double>& activ, int index) {
			activations[index] = activ;
		}

	};
	template<typename T>
	matrice::Matrix<T> feed_pass(const matrice::Matrix<T>& lhs, const matrice::Matrix<double>& rhs) {
		return lhs.mult(rhs);
	}
	
	template <typename T>
	matrice::Matrix<double> feed_forward(Network& network, const matrice::Matrix<T>& input) {
		auto activ = input;
		for (int i = 0; i < network.get_layer_num(); ++i) {
			auto z = feed_pass(network.get_weight(i), activ);
			network.assign_z(z, i);
			activ = sigmoid(z);
			network.assign_activ(activ, i);
		}
		return activ;
	}
}


int main()
{
	std::vector<int> layers{ 2, 2, 1 };
	nn::Network network(layers);
	matrice::Matrix<double> input(layers[0], 1, 2);
	print_matrix(input);
	print_matrix(network.get_weight(0));
	auto res = nn::feed_pass(network.get_weight(0), input);
	print_matrix(res);

	auto y = nn::feed_forward(network, input);
	print_matrix(y);
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
