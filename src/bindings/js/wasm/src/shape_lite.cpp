#include <iostream>

#include "openvino/openvino.hpp"
#include "../include/shape_lite.h" 

ShapeLite::ShapeLite(uintptr_t data, int dim) {
  this->data = data;
  this->dim = dim;
}

uintptr_t ShapeLite::get_data() {
  return this->data;
}

int ShapeLite::get_dim() {
  return this->dim;
}
