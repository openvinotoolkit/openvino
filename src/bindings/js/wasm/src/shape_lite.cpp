// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>

#include "openvino/openvino.hpp"
#include "../include/shape_lite.h"

ShapeLite::ShapeLite(uintptr_t data, int dim) {
    size_t* data_array = reinterpret_cast<size_t*>(data);

    for (int i = 0; i < dim; i++) {
        this->shape.push_back(data_array[i]);
    }
}

ShapeLite::ShapeLite(ov::Shape* shape) {
    this->shape = ov::Shape(*shape);
}

uintptr_t ShapeLite::get_data() {
    return uintptr_t(&this->shape[0]);
}

int ShapeLite::get_dim() {
    return this->shape.size();
}

int ShapeLite::shape_size() {
    return ov::shape_size(this->shape);
}

ov::Shape ShapeLite::get_original() {
    return this->shape;
}
