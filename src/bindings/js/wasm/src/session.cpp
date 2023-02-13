#include <iostream>
#include <limits.h>

#include "openvino/openvino.hpp"

#include "../include/helpers.h"
#include "../include/session.h"

// FIXME: Debug fn, to delete
void printTensor(ov::Tensor t, ov::element::Type type) {
	int size = t.get_size();

	auto data_tensor = reinterpret_cast<uint8_t*>(t.data(type)); 

	for (int i = 0; i < size; i++) { 
		std::cout << +data_tensor[i] << " ";
	} 
	std::cout << std::endl; 
}

Session::Session(std::string xml_path, std::string bin_path, std::string shape, std::string layout) {
	std::cout << "=== 1" << std::endl;
	auto model = loadModel(xml_path, bin_path);
	std::cout << "=== 2" << std::endl;
	try {
		this->model = compileModel(model, shape, layout);
	} catch(std::exception e) {
		std::cout << "== Was error: " << e.what() << std::endl;
	}
	std::cout << "=== 3" << std::endl;
	this->shape = shape;
}

uintptr_t Session::run(uintptr_t inputBuffer, int size) {
	uint8_t* input_data_array = reinterpret_cast<uint8_t *>(inputBuffer);
	std::cout << "=== 123" << std::endl;
	ov::Tensor input_tensor = ov::Tensor(ov::element::u8, this->shape, input_data_array);
	std::cout << "=== 333" << std::endl;
	ov::Tensor output_tensor = performInference(this->model, input_tensor);
	std::cout << "=== 444" << std::endl;
	int output_tensor_size = output_tensor.get_size();
	std::cout << "== Output tensor size: " << output_tensor_size << std::endl;
	this->output_tensor_size = output_tensor_size;

	auto data_tensor = reinterpret_cast<float*>(output_tensor.data(ov::element::f32)); 
	float values[output_tensor_size];

	for (int i = 0; i < output_tensor_size; i++) { 
		values[i] = data_tensor[i];
	}

	return uintptr_t(&values[0]);
}
