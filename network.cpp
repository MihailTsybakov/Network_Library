#include "network.h"

network::network(create_instructor ci)
{
	int prevlayer_size = ci.perception_neurons;
	for (auto layer_size : ci.hidden_layers) layers.push_back(layer(layer_size, prevlayer_size)), prevlayer_size = layer_size;
	layers.push_back(layer(ci.output_neurons, prevlayer_size));
	activation = ci.activation;
	d_activation = ci.d_activation;
	cost_f = ci.cf;
}

void network::save(std::string network_name, bool logs) const
{
	for (size_t i = 0; i < layers.size(); ++i)
	{
		std::stringstream ss;
		ss << network_name << "_" << i << ".txt";
		layers[i].store(ss.str());
	}
	if (logs) std::cout << "<logs> Network stored." << std::endl;
}

void network::load(std::string network_name, int layer_num, bool logs)
{
	layers.resize(layer_num);
	for (size_t i = 0; i < layers.size(); ++i)
	{
		std::stringstream ss;
		ss << network_name << "_" << i << ".txt";
		layers[i].load(ss.str());
	}
	if (logs) std::cout << "<logs> Network loaded." << std::endl;
}

/* xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx */

std::vector<double> network::feedforward(std::vector<double> input)
{
	for (auto layer : layers) input = apply(layer.weights.mult_by_v(input) + layer.bias, activation);
	return input;
}

void network::sgd_step(sgd_instructor si)
{
	std::vector<std::vector<double>> activations, inputs, deltas;
	std::vector<double> layer_perception = si.input;
	for (auto layer : layers) // feeding forward
	{
		std::vector<double> curr_layer_input = layer.weights.mult_by_v(layer_perception) + layer.bias;
		inputs.push_back(curr_layer_input);
		activations.push_back(apply(curr_layer_input, activation));
		layer_perception = apply(curr_layer_input, activation);
	}
	int layer_num = static_cast<int>(layers.size());
	deltas.resize(layer_num);
	deltas[layer_num - 1] = cost_f->lastlayer_delta(activations[layer_num - 1], si.output, inputs[layer_num - 1], d_activation);

	for (int i = layer_num - 2; i >= 0; --i) // backpropagation
	{
		deltas[i] = hadamard_product(apply(inputs[i], d_activation), layers[i + 1].weights.transpose().mult_by_v(deltas[i + 1]));
	}
	layer_perception = si.input;
	for (int i = 0; i < layer_num; ++i) // weigths & bias upd
	{
		if (si.mtx != nullptr) si.mtx->lock();
		for (size_t j = 0; j < layers[i].layer_size; ++j) layers[i].bias[j] -= (si.learning_rate / si.batch_size) * deltas[i][j];
		for (size_t p = 0; p < layers[i].layer_size; ++p)
		{
			for (size_t q = 0; q < layers[i].prevlayer_size; ++q)
			{
				layers[i].weights.M[p][q] = (1 - si.learning_rate * si.L2_lambda) * layers[i].weights.M[p][q]
					- (si.learning_rate / si.batch_size) * deltas[i][p] * layer_perception[q];
			}
		}
		if (si.mtx != nullptr) si.mtx->unlock();
		layer_perception = activations[i];
	}
}

void network::unwrapped_sgd_learn(learn_instructor& li)
{
	time_t seed;

	(li.gen_seed == 0) ? seed = time(nullptr) : seed = static_cast<time_t>(li.gen_seed);

	srand(seed);
	int epoch_upd = 0, patience_limit = 0;
	double best_reached_score = 0.0;

	std::vector<matrix> prev_weights;
	std::vector<std::vector<double>> prev_biases;
	prev_weights.resize(layers.size());
	prev_biases.resize(layers.size());

	for (int epoch = 0; epoch < li.epoch_count; ++epoch)
	{
		epoch_upd++;
		if (li.dynamic_LR == true && epoch_upd >= li.upd_frequency)
		{
			epoch_upd = 0;
			li.learning_rate *= li.mult_factor;
		}
		if (li.epoch_logs == true)
		{
			if (li.mtx != nullptr) li.mtx->lock();
			std::cout << "<logs> [Thread " << std::this_thread::get_id() << "]: Epoch " << epoch + 1 << " started." << std::endl;
			if (li.batch_test == true)
			{
				std::cout << "<logs> [Thread " << std::this_thread::get_id() << "]: Accuracy is " << batch_test(li.validate_input,
					li.validate_output, li.comparator) << "." << std::endl;
			}
			if (li.mtx != nullptr) li.mtx->unlock();
		}
		for (int batch = 0; batch < static_cast<int>(li.train_input->size()) / li.batch_size; ++batch)
		{
			for (int sample = 0; sample < li.batch_size; ++sample)
			{
				int rand_index = rand() % li.train_output.size();

				sgd_instructor si;
				si.batch_size = li.batch_size;
				si.L2_lambda = li.L2_lambda / li.train_input->size();
				si.learning_rate = li.learning_rate;
				si.mtx = li.mtx;
				si.input = (*li.train_input)[rand_index];
				si.output = li.train_output[rand_index];

				sgd_step(si);
			}
		}
		patience_limit++;
		if (li.only_best_score == true)
		{
			double accuracy_score_validation = batch_test(li.validate_input, li.validate_output, li.comparator);
			if (accuracy_score_validation > best_reached_score)
			{
				patience_limit = 0;
				// Updating weights:
				for (size_t i = 0; i < layers.size(); ++i) prev_weights[i] = layers[i].weights, prev_biases[i] = layers[i].bias;
				best_reached_score = accuracy_score_validation;
			}
			else if (patience_limit >= li.patience)
			{
				patience_limit = 0;
				// Returning to previous weights:
				for (size_t i = 0; i < layers.size(); ++i) layers[i].weights = prev_weights[i], layers[i].bias = prev_biases[i];
				if (li.epoch_logs == true)
				{
					std::cout << "<logs> [Thread " << std::this_thread::get_id() << "]: Patience limit exceeded. Returning to better weights." << std::endl;
				}
			}
		}
	}
}

double network::batch_test(std::vector<std::vector<double>> inp,
	std::vector<std::vector<double>> ans,
	std::function<bool(std::vector<double>, std::vector<double>)> comparator)
{
	int correct_answers = 0;
	for (size_t i = 0; i < inp.size(); ++i)
	{
		if (comparator(feedforward(inp[i]), ans[i]) == true) correct_answers++;
	}
	return static_cast<double>(correct_answers) / inp.size();
}

void network::sgd_learn(learn_instructor li)
{
	double* best_score = new double;
	(*best_score) = 0.0;
	std::mutex* mtx = new std::mutex;
	std::vector<std::thread> thread_pool;
	if (li.epoch_logs == true) std::cout << "<logs> Invoking thread pool." << std::endl;

	li.epoch_count /= li.thread_number;
	li.mtx = mtx;
	//li.best_score = best_score;
	if (li.L2_regularize == false)
	{
		li.L2_lambda = 0.0;
	}
	else
	{
		li.L2_lambda /= li.train_input->size();
	}
	if (li.thread_number == 1) li.mtx = nullptr;

	for (int i = 0; i < li.thread_number; ++i) thread_pool.push_back(std::thread(&network::unwrapped_sgd_learn, this, std::ref(li)));
	for (auto th = thread_pool.begin(); th != thread_pool.end(); ++th) th->join();

	if (li.epoch_logs == true) std::cout << "<logs> Thread pool joined." << std::endl;

	delete mtx;
	delete best_score;
}

std::vector<double> quadratic_cost::lastlayer_delta(std::vector<double> pred,
	std::vector<double> ans,
	std::vector<double> inp,
	std::function<double(double)> d_act)
{
	return hadamard_product((pred - ans), apply(inp, d_act));
}

double quadratic_cost::error(std::vector<double> pred, std::vector<double> ans)
{
	double total_error = 0.0;
	for (size_t i = 0; i < pred.size(); ++i)
	{
		total_error += pow((pred[i] - ans[i]), 2);
	}
	return total_error / 2;
}

std::vector<double> cross_entropy_cost::lastlayer_delta(std::vector<double> pred,
	std::vector<double> ans,
	std::vector<double> inp,
	std::function<double(double)> d_act)
{
	return (pred - ans);
}

double cross_entropy_cost::error(std::vector<double> pred, std::vector<double> ans)
{
	double total_error = 0.0;
	for (size_t i = 0; i < pred.size(); ++i)
	{
		total_error -= (ans[i] * log(pred[i]) + (1 - ans[i]) * log(1 - pred[i]));
	}
	return total_error;
}

std::vector<double> gridsearch(gridsearch_params gp, bool logs)
{
	std::vector<double> best_params;
	double best_score = 0.0;
	for (size_t i = 0; i < gp.mult_factor_variations.size(); ++i)
	{
		if (logs) std::cout << "<logs> Mult factor variation: " << i + 1 << std::endl;
		for (size_t j = 0; j < gp.learning_rate_variations.size(); ++j)
		{
			if (logs) std::cout << "<logs> LR variation: " << j + 1 << std::endl;
			for (size_t k = 0; k < gp.l2_lambda_variations.size(); ++k)
			{
				if (logs) std::cout << "<logs> Lambda variation: " << k + 1 << std::endl;
				learn_instructor li;
				// ==================
				li.batch_size = 10;
				li.batch_test = false;
				//li.best_score = nullptr;
				li.comparator = gp.comparator;
				li.dynamic_LR = true;
				li.epoch_count = 10;
				li.epoch_logs = false;
				li.gen_seed = 42;
				li.L2_lambda = gp.l2_lambda_variations[k];
				li.L2_regularize = true;
				li.learning_rate = gp.learning_rate_variations[j];
				li.mtx = nullptr;
				li.mult_factor = gp.mult_factor_variations[i];
				li.only_best_score = false;
				li.thread_number = 1;
				li.train_input = gp.train_input;
				li.train_output = gp.train_output;
				li.upd_frequency = 3;
				li.validate_input = gp.valid_input;
				li.validate_output = gp.valid_output;
				// ==================

				network tmp_net(gp.ci);
				tmp_net.sgd_learn(li);
				double reached_accuracy = tmp_net.batch_test(gp.valid_input, gp.valid_output, gp.comparator);
				if (reached_accuracy > best_score)
				{
					best_score = reached_accuracy;
					best_params = { gp.mult_factor_variations[i], gp.learning_rate_variations[j], gp.l2_lambda_variations[k] };
				}
			}
		}
	}
	if (logs) std::cout << "<logs> GridSearch: best reached accuracy is " << best_score << "." << std::endl;
	return best_params;
}