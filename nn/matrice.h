#ifndef MATRICE_H
#define MATRICE_H
#include <cassert>
#include <vector>

namespace matrice {
	template <typename T>
	class Matrix {
	private:
		std::vector<std::vector<T>> data;
		int sz1, sz2;
	public:
		Matrix(int sz_1, int sz_2, T val) :sz1{ sz_1 }, sz2{sz_2} {
			data = std::vector<std::vector<T>>(sz_1, std::vector<T>(sz_2, val));
		}

		int get_row_num() const { return sz1; }
		int get_col_num() const { return sz2; }
		T get_val(int i, int j) const { 
			assert((0 <= i && i < sz1 && 0 <= j && j < sz2) && "indice are not valid");
			return data[i][j]; 
		}
		std::vector<std::vector<T>> get_data() const { return data; }

		template<typename T>
		void assign_val(int row, int col, T val) { data[row][col] = val; }

		Matrix<T> mult(const Matrix& rhs) const {
			assert((sz2 == rhs.sz1) && "Matrix dimensions doesn't match (mult)");
			Matrix<T> res(sz1, rhs.sz2, 0.0);
			for (int i = 0; i < sz1; ++i) {
				for (int j = 0; j < rhs.sz2; ++j) {
					T val{ 0 };
					for (int z = 0; z < sz2; ++z) {
						val += data[i][z] * rhs.data[z][j];
					}
					res.data[i][j] = val;
				}
			}
			return res;
		}
	};

	Matrix<double> operator*(const Matrix<double>& lhs, const Matrix<double>& rhs) {
		assert((lhs.get_row_num() == rhs.get_row_num()) && (lhs.get_col_num() == rhs.get_col_num()) && \
			("Matrix dimension don't match"));
		Matrix<double> res(lhs.get_row_num(), lhs.get_col_num(), 0.0);
		for (int i = 0; i < lhs.get_row_num(); ++i) {
			for (int j = 0; j < lhs.get_col_num(); ++j) {
				double val = lhs.get_val(i, j) * rhs.get_val(i, j);
				res.assign_val(i, j, val);
			}
		}
		return res;
	}

	Matrix<double> operator/(const Matrix<double>& lhs, double rhs) {
		Matrix<double> res(lhs.get_row_num(), lhs.get_col_num(), 0.0);
		for (int i = 0; i < lhs.get_row_num(); ++i) {
			for (int j = 0; j < lhs.get_col_num(); ++j) {
				double val = lhs.get_val(i, j) / rhs;
				res.assign_val(i, j, val);
			}
		}
		return res;
	}

	Matrix<double> operator>(const Matrix<double> in, double val) {
		Matrix<double> res(in.get_row_num(), in.get_col_num(), 0.0);
		for (int i = 0; i < in.get_row_num(); ++i) {
			for (int j = 0; j < in.get_col_num(); ++j) {
				double temp = static_cast<double>(in.get_val(i, j) > val);
				res.assign_val(i, j, temp);
			}
		}
		return res;
	}
}

#endif //end of MATRICE_H
