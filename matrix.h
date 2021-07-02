#ifndef MATRIX
#define MATRIX

#include <vector>

#include "C:\Users\mihai\Desktop\progy\C & C++\Network\Network\aux_functions.h"

/// Matrix [rows x cols] containing double values
class matrix
{
public:
	size_t rows;
	size_t cols;
	/// Matrix itself
	std::vector<std::vector<double>> M;

	matrix();
	matrix(size_t rows, size_t cols);

	/// Returns transposed version of current matrix
	matrix transpose() const;
	/// Fills matrix with random values from normal gaussian distribution
	void random_fill(double mean, double sigma);
	/// Resizes a matrix to a given shape
	void resize(size_t rows, size_t cols);
	/// Prints matrix
	void print() const;
	/// Multiplicates a matrix by a vector: M x V
	std::vector<double> mult_by_v(std::vector<double> v);
};

#endif//MATRIX