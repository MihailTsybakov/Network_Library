#ifndef AUXF
#define AUXF

#include <vector>
#include <random>
#include <iostream>
#include <ctime>
#include <cmath>
#include <functional>
#include <thread>
#include <mutex>

#include "C:\Users\mihai\Desktop\progy\C & C++\Network\Network\exceptions.h"

/// Scalar product of two vectors
double scalar_product(std::vector<double> v1, std::vector<double> v2);
/// Returns random number generated with normal gaussian distribution
double random_number(double mean, double sigma);
/// Sigmoid activation
double sigmoid(double value);
/// Sigmoid's derivative
double d_sigmoid(double value);


/// Returns a vector of random values from normal distribution
std::vector<double> random_vector(double mean, double sigma, int size);
/// Apllies function to a vector
std::vector<double> apply(std::vector<double> v, std::function<double(double)> f);
/// Hadamard product of two vectors
std::vector<double> hadamard_product(std::vector<double> v1, std::vector<double> v2);
/// Operator+
std::vector<double> operator+(std::vector<double> v1, std::vector<double> v2);
/// Operator-
std::vector<double> operator-(std::vector<double> v1, std::vector<double> v2);
/// Operator*
std::vector<double> operator*(std::vector<double> v, double d);

/// Prints a vector
void print_v(std::vector<double> v);


#endif//AUXF
