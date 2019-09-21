#include "nn.h"
#include "constants.h"
#include <cassert>
#include <cmath>

// implement classes for neural net computation
namespace nn {
	template <class T>
	double sigmoid(T val) { return 1 / (1 + std::pow(constants::euler, -val)); }


	/*double log_loss(int label, double act) {
		assert(0.0 <= act && act <= 1.0 && "act should be between 0 and 1");
		return -1.0 * ((label)* std::log(label) + (1 - label) * std::log(1 - label));
	}*/
}
