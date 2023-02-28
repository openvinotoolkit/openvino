// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <iostream>
#include <stdio.h>

#include "openvino/openvino.hpp"
#include "./shape_lite.h"

class TensorLite {
public:
  TensorLite(std::string type_str, uintptr_t data_buffer, ShapeLite* shape);
  TensorLite(ov::element::Type type, uintptr_t data_buffer, ShapeLite* shape);
  TensorLite(ov::Tensor* tensor);

  uintptr_t get_data();
  ShapeLite* get_shape();
  std::string get_precision();
  std::vector<float> get_vector();
private:
  ov::element::Type type;
  ShapeLite* shape;
  std::vector<float> tensor;
};
