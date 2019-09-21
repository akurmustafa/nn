#ifndef CONSTANTS_H
#define CONSTANTS_H
// constants
namespace nn {
	template <class T>
	double sigmoid(T val);
	double log_loss(int label, double act);
}

#endif
