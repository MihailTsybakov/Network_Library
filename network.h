#ifndef NETWORK
#define NETWORK

#include "C:\Users\mihai\Desktop\progy\C & C++\Network\Network\layer.h"
#include "C:\Users\mihai\Desktop\progy\C & C++\Network\Network\instructors.h"
#include "C:\Users\mihai\Desktop\progy\C & C++\Network\Network\aux_functions.h"
#include "C:\Users\mihai\Desktop\progy\C & C++\Network\Network\matrix.h"

class network
{
private:
	std::vector<layer> layers;
	std::function<double(double)> activation;
	std::function<double(double)> d_activation;
	cost_function* cost_f;
	/// Stohastic gradient descent step
	void sgd_step(sgd_instructor si);
	void unwrapped_sgd_learn(learn_instructor& li);
public:
	network(create_instructor ci);
	void save(std::string network_name, bool logs) const;
	void load(std::string network_name, int layer_num, bool logs);
	/// Feeds input forward through net and spits a result as a vector of doubles
	std::vector<double> feedforward(std::vector<double> input);
	/// Returns network's accuracy on given batch
	double batch_test(std::vector<std::vector<double>> inp,
		std::vector<std::vector<double>> ans,
		std::function<bool(std::vector<double>, std::vector<double>)> comparator);
	/// Trains network with stohastic gradient descent
	void sgd_learn(learn_instructor li);
};

class quadratic_cost : public cost_function
{
	std::vector<double> lastlayer_delta(std::vector<double> pred,
		std::vector<double> ans,
		std::vector<double> inp,
		std::function<double(double)> d_act) override;
	double error(std::vector<double> pred, std::vector<double> ans) override;
};

class cross_entropy_cost : public cost_function
{
	std::vector<double> lastlayer_delta(std::vector<double> pred,
		std::vector<double> ans,
		std::vector<double> inp,
		std::function<double(double)> d_act) override;
	double error(std::vector<double> pred, std::vector<double> ans) override;
};

std::vector<double> gridsearch(gridsearch_params gp, bool logs);

#endif//NETWORK
