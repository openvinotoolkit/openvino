// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "openvino/op/util/arithmetic_reductions_keep_dims.hpp"
#include "openvino/op/util/logical_reduction_keep_dims.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace shape_of {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Node* op, std::vector<TShape> input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1);
    const auto& input_shape = input_shapes[0];
    const auto& input_rank = input_shape.rank();

    auto output_shapes = std::vector<TRShape>(1);

    if (input_rank.is_static()) {
        if (input_shape.size()) {
            output_shapes[0].emplace_back(input_shape.size());
        }
    } else {
        output_shapes[0] = PartialShape::dynamic();
    }
    return output_shapes;
}
}  // namespace shape_of

namespace v0 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const ShapeOf* op, const std::vector<TShape>& input_shapes) {
    return shape_of::shape_infer(op, input_shapes);
}
}  // namespace v0

namespace v3 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const ShapeOf* op, const std::vector<TShape>& input_shapes) {
    return shape_of::shape_infer(op, input_shapes);
}
}  // namespace v3
}  // namespace op
}  // namespace ov
