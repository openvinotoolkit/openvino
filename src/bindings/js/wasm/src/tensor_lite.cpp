#include <stdio.h>
#include <iostream>

#include "openvino/openvino.hpp"

#include "../include/tensor_lite.h"
#include "../include/shape_lite.h"

TensorLite::TensorLite(ov::element::Type type, uintptr_t data_buffer, ShapeLite* shape) {
  this->type = type;
  this->data_buffer = data_buffer;
  this->shape = shape;
}

TensorLite::TensorLite(std::string typeStr, uintptr_t data_buffer, ShapeLite* shape) {
  this->type = ov::element::f16;
  this->data_buffer = data_buffer;
  this->shape = shape;
}

ShapeLite* TensorLite::get_shape() {
  // std::string s = "float16";
  // std::cout << "= Val: " << this->type << std::endl;

  return this->shape;
}

uintptr_t TensorLite::get_data() {
  return this->data_buffer;
}

std::string get_precision() {
  std::string s = "float16";

  return s + "123";
}
