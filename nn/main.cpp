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
	template <typename T>
	double leaky_relu(T in, double alpha = 0.01) {
		return in > 0 ? in, alpha * in;
	}

	template <typename T>
	matrice::Matrix<double> leaky_relu(const matrice::Matrix<T>& in) {
		matrice::Matrix<double> out(in.get_row_num(), in.get_col_num(), 0.0);
		for (int i = 0; i < out.get_row_num(); ++i) {
			for (int j = 0; j < out.get_col_num(); ++j) {
				double val = leaky_relu(in.get_val(i, j));
				out.assign_val(i, j, val);
			}
		}
		return out;
	}

	template <typename T>
	double d_leaky_relu(T in, double alpha = 0.01) {
		return in > 0 ? 1, alpha;
	}

	template <typename T>
	matrice::Matrix<double> d_leaky_relu(const matrice::Matrix<T>& in) {
		matrice::Matrix<double> out(in.get_row_num(), in.get_col_num(), 0.0);
		for (int i = 0; i < out.get_row_num(); ++i) {
			for (int j = 0; j < out.get_col_num(); ++j) {
				double val = d_leaky_relu(in.get_val(i, j));
				out.assign_val(i, j, val);
			}
		}
		return out;
	}

	template <typename T>
	T relu(T in) {
		return in > 0 ? in : 0;
	}

	template <typename T>
	matrice::Matrix<T> relu(const matrice::Matrix<T>& in) {
		matrice::Matrix<T> out(in.get_row_num(), in.get_col_num(), 0.0);
		for (int i = 0; i < out.get_row_num(); ++i) {
			for (int j = 0; j < out.get_col_num(); ++j) {
				T val = relu(in.get_val(i, j));
				out.assign_val(i, j, val);
			}
		}
		return out;
	}
	
	template <typename T>
	T d_relu(T in) {
		return in > 0 ? 1 : 0;
	}

	template <typename T>
	matrice::Matrix<T> d_relu(const matrice::Matrix<T>& in) {
		matrice::Matrix<T> out(in.get_row_num(), in.get_col_num(), 0.0);
		for (int i = 0; i < out.get_row_num(); ++i) {
			for (int j = 0; j < out.get_col_num(); ++j) {
				T val = d_relu(in.get_val(i, j));
				out.assign_val(i, j, val);
			}
		}
		return out;
	}

	template <typename T>
	double tanh(T in) {
		return 2.0 / (1 + std::pow(constants::euler, -2 * in)) - 1.0;
	}

	template <typename T>
	matrice::Matrix<double> tanh(const matrice::Matrix<T>& in) {
		matrice::Matrix<double> out(in.get_row_num(), in.get_col_num(), 0.0);
		for (int i = 0; i < out.get_row_num(); ++i) {
			for (int j = 0; j < out.get_col_num(); ++j) {
				double val = tanh(in.get_val(i, j));
				out.assign_val(i, j, val);
			}
		}
		return out;
	}

	template <typename T>
	double d_tanh(T in) {
		return (1.0 -std::pow(tanh(in), 2));
	}

	template <typename T>
	matrice::Matrix<double> d_tanh(const matrice::Matrix<T>& in) {
		matrice::Matrix<double> out(in.get_row_num(), in.get_col_num(), 0.0);
		for (int i = 0; i < out.get_row_num(); ++i) {
			for (int j = 0; j < out.get_col_num(); ++j) {
				double val = d_tanh(in.get_val(i, j));
				out.assign_val(i, j, val);
			}
		}
		return out;
	}

	template<typename T>
	double sigmoid(T in) {
		double out = 1 / (1 + std::pow(constants::euler, -in));
		return out;
	}

	template <typename T>
	matrice::Matrix<T> sigmoid(const matrice::Matrix<T>& in) {
		matrice::Matrix<double> out(in.get_row_num(), in.get_col_num(), 0.0);
		for (int i = 0; i < out.get_row_num(); ++i) {
			for (int j = 0; j < out.get_col_num(); ++j) {
				double val = sigmoid(in.get_val(i, j));
				out.assign_val(i, j, val);
			}
		}
		return out;
	}

	template<typename T>
	double d_sigmoid(T in) {
		double res = sigmoid(in);
		return res * (1.0 - res);
	}

	template <typename T>
	matrice::Matrix<double> d_sigmoid(const matrice::Matrix<T>& in) {
		matrice::Matrix<double> out(in.get_row_num(), in.get_col_num(), 0.0);
		for (int i = 0; i < out.get_row_num(); ++i) {
			for (int j = 0; j < out.get_col_num(); ++j) {
				double val = d_sigmoid(in.get_val(i, j));
				out.assign_val(i, j, val);
			}
		}
		return out;
	}

	double cross_entropy(double label, double activ) {
		double res{ 0.0 };
		if (label == 1)
			res = -std::log(activ);
		else if (label == 0)
			res = -std::log(1.0 - activ);
		else {
			res = -1.0 * (std::log(activ) + std::log(1.0 - activ));
		}
		return res;
	}

	matrice::Matrix<double> cross_entropy(const matrice::Matrix<double> labels, const matrice::Matrix<double> activs) {
		assert((labels.get_row_num() == activs.get_row_num()) && (labels.get_col_num() == activs.get_col_num()) \
			&& "dimension don't match");
		matrice::Matrix<double> loss(labels.get_row_num(), labels.get_col_num(), 0.0);
		for (int i = 0; i < labels.get_row_num(); ++i) {
			for (int j = 0; j < labels.get_col_num(); ++j) {
				double val = cross_entropy(labels.get_val(i, j), activs.get_val(i, j));
				loss.assign_val(i, j, val);
			}
		}
		return loss;
	}


	double d_cross_entropy(double label, double activ) {
		return -1.0 * (label / activ - (1.0 - label) / (1.0 - activ));
	}

	matrice::Matrix<double> d_cross_entropy(const matrice::Matrix<double> labels, const matrice::Matrix<double> activs) {
		assert((labels.get_row_num() == activs.get_row_num()) && (labels.get_col_num() == activs.get_col_num()) \
			&& "dimension don't match");
		matrice::Matrix<double> loss(labels.get_row_num(), labels.get_col_num(), 0.0);
		for (int i = 0; i < labels.get_row_num(); ++i) {
			for (int j = 0; j < labels.get_col_num(); ++j) {
				double val = d_cross_entropy(labels.get_val(i, j), activs.get_val(i, j));
				loss.assign_val(i, j, val);
			}
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
	matrice::Matrix<T> forward(const matrice::Matrix<T>& lhs, const matrice::Matrix<double>& rhs) {
		return lhs.mult(rhs);
	}
	
	template <typename T>
	matrice::Matrix<double> feed_forward(Network& network, const matrice::Matrix<T>& input) {
		auto activ = input;
		for (int i = 0; i < network.get_layer_num(); ++i) {
			auto z = forward(network.get_weight(i), activ);
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
