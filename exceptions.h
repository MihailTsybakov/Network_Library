#ifndef NETWORK_EXC
#define NETWORK_EXC

#include <exception>
#include <string>

class aux_f_exception : public std::exception
{
private:
	std::string error_msg;
public:
	aux_f_exception(std::string error_msg) { this->error_msg = error_msg; }
	const char* what() const noexcept override { return error_msg.c_str(); }
};

class layer_exception : public std::exception
{
private:
	std::string error_msg;
public:
	layer_exception(std::string error_msg) { this->error_msg = error_msg; }
	const char* what() const noexcept override { return error_msg.c_str(); }
};

class matrix_exception : public std::exception
{
private:
	std::string error_msg;
public:
	matrix_exception(std::string error_msg) { this->error_msg = error_msg; }
	const char* what() const noexcept override { return error_msg.c_str(); }
};

class network_exception : public std::exception
{
private:
	std::string error_msg;
public:
	network_exception(std::string error_msg) { this->error_msg = error_msg; }
	const char* what() const noexcept override { return error_msg.c_str(); }
};

#endif//NETWORK_EXC
