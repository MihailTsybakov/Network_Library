#include "network.h"
#pragma comment(lib, "neural_network.lib")
#include "helper_funcs.h"

int main()
{
	// ================================================================
	cost_function* quad_cost = new quadratic_cost;
	cost_function* cross_cost = new cross_entropy_cost;
	create_instructor ci;
	ci.cf = cross_cost;
	ci.activation = sigmoid;
	ci.d_activation = d_sigmoid;
	ci.perception_neurons = 28 * 28;
	ci.output_neurons = 10;
	ci.hidden_layers = { 350 };
	// ================================================================

	network N(ci);

	mnist_reader mr_train("C:\\Users\\mihai\\Desktop\\progy\\C & C++\\Digits_Identifier\\Data", "train_labels.txt", "train-images.idx3-ubyte");
	mnist_reader mr_test("C:\\Users\\mihai\\Desktop\\progy\\C & C++\\Digits_Identifier\\Data", "test_labels.txt", "t10k-images.idx3-ubyte");

	std::vector<int> train_labels;
	std::vector<std::vector<double>> train_input;

	append_labels(train_labels, "C:\\Users\\mihai\\Desktop\\progy\\Additional\\Elastic_Distortions\\train_labels.txt", 60'000);
	append_labels(train_labels, "C:\\Users\\mihai\\Desktop\\progy\\Additional\\Elastic_Distortions\\train_labels.txt", 60'000);
	//append_labels(train_labels, "C:\\Users\\mihai\\Desktop\\progy\\Additional\\Elastic_Distortions\\train_labels.txt", 60'000); 

	append_from_binary(train_input, "C:\\Users\\mihai\\Desktop\\progy\\Additional\\Elastic_Distortions\\Basic_Bin.bin", 60'000);    // Basic Set
	append_from_binary(train_input, "C:\\Users\\mihai\\Desktop\\progy\\Additional\\Elastic_Distortions\\Extended_Bin.bin", 60'000); // Extended Set
	//append_from_binary(train_input, "C:\\Users\\mihai\\Desktop\\progy\\Additional\\Elastic_Distortions\\Extended_1_Bin.bin", 60'000); // Extended Set #2

	//train_input = form_perception(mr_train.read_images(60'000));
	std::vector<std::vector<double>> train_output = form_answers(train_labels);
	std::vector<std::vector<double>> tval_input = form_perception(mr_test.read_images(2'000));
	std::vector<std::vector<double>> tval_output = form_answers(mr_test.read_labels(2'000));
	std::vector<std::vector<double>> test_input, test_output, validate_input, validate_output;

	// Test-Validate split
	for (int i = 0; i < 2'000; ++i)
	{
		if (i < 1'000)
		{
			test_input.push_back(tval_input[i]);
			test_output.push_back(tval_output[i]);
		}
		else
		{
			validate_input.push_back(tval_input[i]);
			validate_output.push_back(tval_output[i]);
		}
	}

	std::cout << "<logs> Data scanned." << std::endl;


	learn_instructor li;
	// ==============================
	li.batch_size = 10;
	li.batch_test = true;
	li.comparator = compare;
	li.dynamic_LR = true;
	li.only_best_score = true;          // Only one-thread feature now
	li.patience = 4;                    // Only one-thread feature now
	li.epoch_count = 16;
	li.epoch_logs = true;
	li.L2_lambda = 5.0;
	li.L2_regularize = true;
	li.learning_rate = 1.5;
	li.mtx = nullptr;
	li.mult_factor = 0.9;
	li.thread_number = 1;
	li.train_input = &train_input;
	li.train_output = train_output;
	li.upd_frequency = 4;
	li.validate_input = validate_input;
	li.validate_output = validate_output;
	li.gen_seed = 0;
	// ==============================

	N.sgd_learn(li);

	N.save("Distortion", false);

	//N.load("Cuda_Learned_Net", 2, true);

	std::cout << "Accuracy on test set: " << N.batch_test(test_input, test_output, compare) << std::endl;

	delete quad_cost;
	delete cross_cost;
	return 0;
}