#ifndef INSTRUCTORS
#define INSTRUCTORS

#include "C:\Users\mihai\Desktop\progy\C & C++\Network\Network\aux_functions.h"

class cost_function
{
public:
	virtual std::vector<double> lastlayer_delta(std::vector<double> pred,
		std::vector<double> ans,
		std::vector<double> inp,
		std::function<double(double)> d_act) = 0;
	virtual double error(std::vector<double> pred, std::vector<double> ans) = 0;
};

typedef struct
{
	int perception_neurons;
	int output_neurons;
	std::function<double(double)> activation;
	std::function<double(double)> d_activation;
	cost_function* cf;
	std::vector<int> hidden_layers;

}   create_instructor;

typedef struct
{
	int batch_size;
	int epoch_count;
	int thread_number;
	int upd_frequency;
	int gen_seed;
	int patience;

	double learning_rate;
	double L2_lambda;
	double mult_factor;

	bool epoch_logs;
	bool batch_test;
	bool dynamic_LR;
	bool L2_regularize;
	bool only_best_score;

	std::vector<std::vector<double>>* train_input;
	std::vector<std::vector<double>> train_output;
	std::vector<std::vector<double>> validate_input;
	std::vector<std::vector<double>> validate_output;
	std::function<bool(std::vector<double>, std::vector<double>)> comparator;

	std::mutex* mtx;

}   learn_instructor;

typedef struct
{
	int batch_size;
	double learning_rate;
	double L2_lambda;
	std::vector<double> input;
	std::vector<double> output;

	std::mutex* mtx;

}   sgd_instructor;

typedef struct
{
	std::vector<double> learning_rate_variations;
	std::vector<double> l2_lambda_variations;
	std::vector<double> mult_factor_variations;

	std::vector<std::vector<double>>* train_input;
	std::vector<std::vector<double>> train_output;
	std::vector<std::vector<double>> valid_input;
	std::vector<std::vector<double>> valid_output;

	create_instructor ci;
	std::function<bool(std::vector<double>, std::vector<double>)> comparator;

}   gridsearch_params;

#endif//INSTRUCTORS
