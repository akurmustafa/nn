#ifndef MATRICE_H
#define MATRICE_H
#include <vector>

namespace matrice {
	template <typename T>
	class Matrix {
	private:
		std::vector<std::vector<T>> data;
	public:
		Matrix(int sz1, int sz2) {
			data = std::vector<std::vector<T>>(sz1, std::vector<T>(sz2));
		}
		template<typename T>
		void set_data(T val, int row, int col) { data[row][col] = val; }
	};
}


#endif //end of MATRICE_H
