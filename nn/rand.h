#ifndef RANDLIB_H
#define RANDLIB_H
#include "matrice.h"
#include <random>

namespace rand_lib {
	std::random_device rd;
	std::mt19937 gen(rd());

	double randn(double mean, double sigma) {
		std::normal_distribution<> dist(mean, sigma);
		return dist(gen);
	}
	matrice::Matrix<double> rand_matrix(int sz1, int sz2, double mean, double sigma) {
		matrice::Matrix<double> res(sz1, sz2, 0.0);
		for (int i = 0; i < sz1; ++i) {
			for (int j = 0; j < sz2; ++j) {
				auto val = randn(mean, sigma);
				res.assign_val(i, j, val);
			}
		}
		return res;
	}
}

#endif // end of RAND_H
