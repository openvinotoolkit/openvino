// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "pooling_shape_inference_util.hpp"
#include "utils.hpp"

namespace ov {
namespace op {

namespace v1 {
template <class TShape>
std::vector<TShape> shape_infer(const MaxPool* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1);
    const auto& data_shape = input_shapes[0];

    const auto dilations = Strides(op->get_kernel().size(), 1);

    pooling::update_and_validate_attributes(const_cast<MaxPool*>(op), data_shape, dilations);

    return {pooling::out_shape_infer(op, data_shape, dilations)};
}

template <class TShape>
void shape_infer(const MaxPool* op, const std::vector<TShape>& input_shapes, std::vector<TShape>& output_shapes) {
    output_shapes = shape_infer(op, input_shapes);
}
}  // namespace v1

namespace v8 {
template <class TShape>
std::vector<TShape> shape_infer(const MaxPool* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1);
    const auto& data_shape = input_shapes[0];

    auto dilations = op->get_dilations();
    if (dilations.empty()) {
        dilations.resize(op->get_kernel().size(), 1);
    }

    pooling::update_and_validate_attributes(const_cast<MaxPool*>(op), data_shape, dilations);

    auto output_shape = pooling::out_shape_infer(op, data_shape, dilations);
    return {2, output_shape};
}

template <class TShape>
void shape_infer(const MaxPool* op, const std::vector<TShape>& input_shapes, std::vector<TShape>& output_shapes) {
    output_shapes = shape_infer(op, input_shapes);
}
}  // namespace v8
}  // namespace op
}  // namespace ov
