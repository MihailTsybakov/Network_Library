#include "matrix.h"

matrix::matrix()
{
	rows = 0; cols = 0;
}

matrix::matrix(size_t rows, size_t cols)
{
	this->rows = rows;
	this->cols = cols;
	M.resize(rows);
	for (size_t row = 0; row < rows; ++row) M[row].resize(cols);
}

matrix matrix::transpose() const
{
	matrix T(cols, rows);
	for (size_t row = 0; row < rows; ++row)
	{
		for (size_t col = 0; col < cols; ++col) T.M[col][row] = this->M[row][col];
	}
	return T;
}

void matrix::random_fill(double mean, double sigma)
{
	for (size_t row = 0; row < rows; ++row)
	{
		for (size_t col = 0; col < cols; ++col) M[row][col] = random_number(mean, sigma);
	}
}

void matrix::resize(size_t rows, size_t cols)
{
	this->rows = rows;
	this->cols = cols;
	M.resize(rows);
	for (size_t row = 0; row < rows; ++row) M[row].resize(cols);
}

void matrix::print() const
{
	for (size_t row = 0; row < rows; ++row)
	{
		for (size_t col = 0; col < cols; ++col) std::cout << M[row][col] << " ";
		std::cout << std::endl;
	}
}

std::vector<double> matrix::mult_by_v(std::vector<double> v)
{
	if (v.size() != cols) throw matrix_exception("Error: dimension mismatch in matrix multiplication.");
	std::vector<double> res;
	res.resize(rows);
	for (size_t i = 0; i < rows; ++i) res[i] = scalar_product(v, M[i]);
	return res;
}