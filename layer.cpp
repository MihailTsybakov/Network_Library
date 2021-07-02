#include "layer.h"

layer::layer()
{
	layer_size = 0;
	prevlayer_size = 0;
	weights = matrix();
	bias = std::vector<double>();
}

layer::layer(size_t layer_size, size_t prevlayer_size)
{
	this->layer_size = layer_size;
	this->prevlayer_size = prevlayer_size;
	weights.resize(layer_size, prevlayer_size);
	weights.random_fill(0, 1 / sqrt(prevlayer_size));
	bias = random_vector(0.0, 1.0, layer_size);
}

void layer::store(std::string filename) const
{
	if (filename.find(".txt") == std::string::npos) throw layer_exception("Error: wrong save format - cannot store layer.");
	std::ofstream layer_file;
	layer_file.open(filename, std::ios::binary);
	if (!layer_file.is_open() == true) throw layer_exception("Error: failed to open dump file.");

	layer_file << layer_size << " " << prevlayer_size << std::endl;

	for (size_t i = 0; i < layer_size; ++i)
	{
		for (size_t j = 0; j < prevlayer_size; ++j) layer_file << weights.M[i][j] << " ";
		layer_file << bias[i] << std::endl;
	}

	layer_file.close();
}

void layer::load(std::string filename)
{
	if (filename.find(".txt") == std::string::npos) throw layer_exception("Error: wrong file format - cannot load layer.");
	std::ifstream layer_file;
	layer_file.open(filename, std::ios::in);
	if (!layer_file.is_open() == true) throw layer_exception("Error: failed to open dump file.");

	layer_file >> layer_size;
	layer_file >> prevlayer_size;

	bias.resize(layer_size);
	weights.resize(layer_size, prevlayer_size);

	for (size_t i = 0; i < layer_size; ++i)
	{
		for (size_t j = 0; j < prevlayer_size; ++j) layer_file >> weights.M[i][j];
		layer_file >> bias[i];
	}

	layer_file.close();
}