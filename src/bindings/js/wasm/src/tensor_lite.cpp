#include <stdio.h>
#include <iostream>

#include "openvino/openvino.hpp"

#include "../include/tensor_lite.h"
#include "../include/shape_lite.h"

std::vector<float> to_vector(uintptr_t data_buffer, int length) {
  std::vector<float> vector;

  float* data_array = reinterpret_cast<float*>(data_buffer);

  for (int i = 0; i < length; i++) {
    vector.push_back(data_array[i]);
  }

  return vector;
}

TensorLite::TensorLite(ov::element::Type type, uintptr_t data_buffer, ShapeLite* shape) {
  this->type = type;
  this->shape = std::shared_ptr<ShapeLite>(shape);
  this->tensor = to_vector(data_buffer, this->shape->shape_size());
}

TensorLite::TensorLite(std::string type_str, uintptr_t data_buffer, ShapeLite* shape) {
  // FIXME: replace hardcoded precision
  this->type = ov::element::f32;
  this->shape = std::shared_ptr<ShapeLite>(shape);
  this->tensor = to_vector(data_buffer, this->shape->shape_size());
}

TensorLite::TensorLite(ov::Tensor* tensor) {
  ov::Shape originalShape = tensor->get_shape();

  int tensor_size = tensor->get_size();
  auto data_tensor = reinterpret_cast<float*>(tensor->data(ov::element::f32)); 

	for (int i = 0; i < tensor_size; i++) { 
		this->tensor.push_back(data_tensor[i]);
	}

  // FIXME: replace hardcoded precision
  this->type = ov::element::f32;
  this->shape = std::shared_ptr<ShapeLite>(new ShapeLite(&originalShape));
}

ShapeLite* TensorLite::get_shape() {
  return this->shape.get();
}

uintptr_t TensorLite::get_data() {
  return uintptr_t(&this->tensor[0]);
}

std::string TensorLite::get_precision() {
  return "float32";
}

std::vector<float> TensorLite::get_vector() {
  return this->tensor;
}
