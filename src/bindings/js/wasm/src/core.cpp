// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
#endif

#include "openvino/openvino.hpp"
#include "../include/helpers.h"
#ifdef __EMSCRIPTEN__
#include "../include/session.h"
#endif
#include "../include/shape_lite.h"
#include "../include/tensor_lite.h"

#ifdef __EMSCRIPTEN__
using namespace emscripten;
#endif

std::string getVersionString() {
	ov::Version version = ov::get_openvino_version();
	std::string str;

	return str.assign(version.buildNumber);
}

std::string getDescriptionString() {
	ov::Version version = ov::get_openvino_version();
	std::string str;

	return str.assign(version.description);
}

std::string readModel(std::string xml_path, std::string bin_path) {
	std::cout << "== Loading" << std::endl;
	auto model = loadModel(xml_path, bin_path);
	std::cout << "== Loaded sucessfully" << std::endl;
	ov::CompiledModel compiled_model = compileModel(model, "[1, 224, 224, 3]", "NHWC");

	auto v = compiled_model.inputs();
	std::cout << "Inputs: " << v.size() << std::endl;

	ov::Tensor t = getRandomTensor();
	ov::Tensor output = performInference(compiled_model, t);

	std::cout << "== Output tensor size: " << output.get_size() << std::endl;
	
	return "Was processed: " + xml_path + " and " + bin_path;
}

void processShape(ShapeLite s) {
	std::cout << "== Process Shape ==" << std::endl;
	std::cout << "== dim: " << s.get_dim() << std::endl;

	auto input_data_array = reinterpret_cast<uint16_t *>(s.get_data());

	std::cout << "== size: " << sizeof(input_data_array) << std::endl;

	for (int i = 0; i < s.get_dim(); i++) {
		std::cout << "== [" << i << "]: " << input_data_array[i] << std::endl;
	}
}

ShapeLite getShape() {
	uint16_t values[3];

	values[0] = 1;
	values[1] = 224;
	values[2] = 224;

	return ShapeLite(uintptr_t(&values[0]), 3);
}

void debugShape(ShapeLite *s) {
	int dim = s->get_dim();
	uintptr_t inputBuffer = s->get_data();
	uint16_t* input_data_array = reinterpret_cast<uint16_t*>(inputBuffer);

	std::cout << "= Debug Shape =" << std::endl;
	std::cout << "== Dim: " << dim << std::endl;

	for (int i = 0; i < dim; i++) {
		std::cout << "== [" << i << "]: " << input_data_array[i] << std::endl;
	}

	std::cout << "= End Debug Shape =" << std::endl;
}

ShapeLite getShape2() {
	ShapeLite s = getShape();
	debugShape(&s);

	return s;
}

TensorLite getTensor() {
	uint16_t values[2];

	values[0] = 2;
	values[1] = 4;

	ShapeLite s = ShapeLite(uintptr_t(&values[0]), 2);

	debugShape(&s);

	float t_values[8];
	t_values[0] = 1.2;
	t_values[1] = 2.3;
	t_values[2] = 3.4;
	t_values[3] = 4.5;
	t_values[4] = 5.6;
	t_values[5] = 6.7;
	t_values[6] = 7.8;
	t_values[7] = 8.9;

	std::string precision = "float16";

	TensorLite t = TensorLite(precision, uintptr_t(&t_values[0]), &s);

	// std::cout << "== Get Tensor ==" << std::endl;
	ShapeLite* shape_in_tensor = t.get_shape();

	debugShape(shape_in_tensor);

	return t;
}

#ifndef __EMSCRIPTEN__
int main() {
	const std::string xml_path = "./data/models/v3-small_224_1.0_float.xml";
	const std::string bin_path = "./data/models/v3-small_224_1.0_float.bin";

	std::cout << "== start" << std::endl;

	std::cout << getVersionString() << std::endl;
	std::cout << getDescriptionString() << std::endl;

	readModel(xml_path, bin_path);

	std::cout << "== end" << std::endl;
}
#endif

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_BINDINGS(my_module) {
	// function("getVersionString", &getVersionString);
	// function("getDescriptionString", &getDescriptionString);
	function("readModel", &readModel);

	class_<Session>("Session")
    .constructor<std::string, std::string, std::string, std::string >()
    .function("run", &Session::run)
		.property("outputTensorSize", &Session::output_tensor_size)
    ;

	class_<ShapeLite>("Shape")
		.constructor<uintptr_t, int>()
		.function("getDim", &ShapeLite::get_dim)
		.function("getData", &ShapeLite::get_data)
		;

	class_<TensorLite>("Tensor")
		// .constructor<std::string, uintptr_t, ShapeLite*, allow_raw_pointers()>()
		.function("getShape", &TensorLite::get_shape, allow_raw_pointers())
		.function("getData", &TensorLite::get_data)
		// .function("getPrecision", &TensorLite::get_precision)
		;

	function("processShape", &processShape);
	function("getShape", &getShape);
	function("getTensor", &getTensor);
	function("getShape2", &getShape2);
}
#endif
