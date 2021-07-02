#include "aux_functions.h"

double scalar_product(std::vector<double> v1, std::vector<double> v2)
{
	if (v1.size() != v2.size()) throw aux_f_exception("Error: dimension mismatch in scalar product.");
	double sp = 0.0;
	for (size_t i = 0; i < v1.size(); ++i) sp += v1[i] * v2[i];
	return sp;
}

double random_number(double mean, double sigma)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<double> nd(mean, sigma);
	return nd(gen);
}

double sigmoid(double value)
{
	return (1 / (1 + exp(-value)));
}

double d_sigmoid(double value)
{
	return sigmoid(value) * (1 - sigmoid(value));
}

std::vector<double> random_vector(double mean, double sigma, int size)
{
	std::vector<double> rv;
	rv.resize(size);
	for (int i = 0; i < size; ++i) rv[i] = random_number(mean, sigma);
	return rv;
}

std::vector<double> apply(std::vector<double> v, std::function<double(double)> f)
{
	std::vector<double> av;
	av.resize(v.size());
	for (size_t i = 0; i < v.size(); ++i) av[i] = f(v[i]);
	return av;
}

std::vector<double> hadamard_product(std::vector<double> v1, std::vector<double> v2)
{
	if (v1.size() != v2.size()) throw aux_f_exception("Error: dimension mismatch in hadamard product.");
	std::vector<double> hp;
	hp.resize(v2.size());
	for (size_t i = 0; i < v1.size(); ++i) hp[i] = v1[i] * v2[i];
	return hp;
}

std::vector<double> operator+(std::vector<double> v1, std::vector<double> v2)
{
	if (v1.size() != v2.size()) throw aux_f_exception("Error: dimension mismatch in operator+.");
	std::vector<double> sum;
	sum.resize(v1.size());
	for (size_t i = 0; i < v1.size(); ++i) sum[i] = v1[i] + v2[i];
	return sum;
}

std::vector<double> operator-(std::vector<double> v1, std::vector<double> v2)
{
	if (v1.size() != v2.size()) throw aux_f_exception("Error: dimension mismatch in operator+.");
	std::vector<double> delta;
	delta.resize(v1.size());
	for (size_t i = 0; i < v1.size(); ++i) delta[i] = v1[i] - v2[i];
	return delta;
}

std::vector<double> operator*(std::vector<double> v, double d)
{
	std::vector<double> mult;
	mult.resize(v.size());
	for (size_t i = 0; i < v.size(); ++i) mult[i] = v[i] * d;
	return mult;
}

void print_v(std::vector<double> v)
{
	std::cout << "[";
	for (size_t i = 0; i < v.size() - 1; ++i) std::cout << v[i] << ", ";
	std::cout << v[v.size() - 1] << "]" << std::endl;
}