#ifndef LAYER
#define LAYER

#include "C:\Users\mihai\Desktop\progy\C & C++\Network\Network\aux_functions.h"
#include "C:\Users\mihai\Desktop\progy\C & C++\Network\Network\matrix.h"
#include "C:\Users\mihai\Desktop\progy\C & C++\Network\Network\exceptions.h"

#include <fstream>

class network;
/// Network's neuron layer
class layer
{
private:
	size_t layer_size;
	size_t prevlayer_size;
	matrix weights;
	std::vector<double> bias;
public:
	layer();
	layer(size_t layer_size, size_t prevlayer_size);
	/// Stores layer's weights and biases into filename.bin
	void store(std::string filename) const;
	/// Loads layer's weights and biases from filename.bin
	void load(std::string filename);
	friend class network;
};

#endif//LAYER
