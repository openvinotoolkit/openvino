// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <limits.h>

#include "openvino/openvino.hpp"

#include "../include/helpers.h"
#include "../include/session.h"
#include "../include/shape_lite.h"

Session::Session(std::string xml_path, std::string bin_path, ShapeLite* shape, std::string layout) {
	auto model = loadModel(xml_path, bin_path);
	try {
		this->model = compileModel(model, shape->get_original(), layout);
	} catch(const std::exception& e) {
		std::cout << "== Error in Session constructor: " << e.what() << std::endl;
		throw e;
	}

	this->shape = shape;
}

TensorLite Session::run(TensorLite* tensor_lite) {
	std::cout << "== Run inference: " << std::endl;
	ov::Tensor output_tensor;

	try {
		output_tensor = performInference(this->model, *tensor_lite->get_tensor());
	} catch(const std::exception& e) {
		std::cout << "== Error in run: " << e.what() << std::endl;
		throw e;
	}

	return TensorLite(output_tensor);
}
