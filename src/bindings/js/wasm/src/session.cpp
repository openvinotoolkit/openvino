#include <iostream>
#include <limits.h>

#include "openvino/openvino.hpp"

#include "../include/helpers.h"
#include "../include/session.h"
#include "../include/shape_lite.h"

Session::Session(std::string xml_path, std::string bin_path, ShapeLite* shape, std::string layout) {
	auto model = loadModel(xml_path, bin_path);
	try {
		this->model = compileModel(model, shape->to_string(), layout);
	} catch(const std::exception& e) {
		std::cout << "== Error in Session constructor: " << e.what() << std::endl;
		throw e;
	}

	this->shape = shape;
}

TensorLite Session::run(TensorLite* tensor) {
	std::vector<float> v = tensor->get_vector();

	// FIXME: this transformation should be automatically processed by Tensor
	std::vector<uint8_t> arr;
	int size = v.size();
	for (int i = 0; i < size; i++) {
		arr.push_back(v[i]);
	}

	ov::Tensor input_tensor = ov::Tensor(ov::element::u8, this->shape->to_string(), &arr[0]);
	ov::Tensor output_tensor = performInference(this->model, input_tensor);

	return TensorLite(&output_tensor);
}
