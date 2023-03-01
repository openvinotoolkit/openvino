// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>

#include "openvino/openvino.hpp"
#include "../include/shape_lite.h" 

ShapeLite::ShapeLite(uintptr_t data, int dim) {
  uint16_t* data_array = reinterpret_cast<uint16_t*>(data);

  for (int i = 0; i < dim; i++) {
    this->shape.push_back(data_array[i]);
  }
}

ShapeLite::ShapeLite(ov::Shape* shape) {
  for (auto d : *shape) {
    this->shape.push_back(d);
  }
}

uintptr_t ShapeLite::get_data() {
  return uintptr_t(&this->shape[0]);
}

int ShapeLite::get_dim() {
  return this->shape.size();
}

int ShapeLite::shape_size() {
  int size = 1;

  for (auto d : this->shape) size *= d;

  return size;
}

std::string ShapeLite::to_string() {
  std::string result = "[";
  int counter = 0;

  for (uint16_t d : this->shape) {
    result = result + std::to_string(d);

    counter++;

    if (counter != this->shape.size()) result = result + ",";
  }

  return result + "]";
}
