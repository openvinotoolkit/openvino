// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/validation_util.hpp"
#include "openvino/op/one_hot.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace util {

template <class T>
struct GetNotNegative {
    const Node* m_op;

    GetNotNegative(const Node* op) : m_op{op} {}

    template <class V>
    T operator()(const V v) const {
        NODE_VALIDATION_CHECK(m_op, cmp::ge(v, 0), "OneHot depth value can't be negative.");
        return static_cast<T>(v);
    }
};
}  // namespace util
namespace v1 {
void inline resolve_axis(OneHot* op) {
    if (op->get_input_size() < 1) {
        return;
    }
    const auto& indices_shape = op->get_input_partial_shape(0);
    if (indices_shape.rank().is_static()) {
        op->m_axis = ov::util::try_normalize_axis(op->m_axis, indices_shape.rank() + 1, *op);
    }
}

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const OneHot* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4);
    using DimType = typename T::value_type;
    const auto& indices_shape = input_shapes[0];
    const auto& depth_shape = input_shapes[1];
    const auto& on_value_shape = input_shapes[2];
    const auto& off_value_shape = input_shapes[3];

    NODE_VALIDATION_CHECK(op,
                          depth_shape.is_dynamic() || ov::is_scalar(depth_shape.to_shape()),
                          "depth input must be scalar.");

    NODE_VALIDATION_CHECK(op,
                          on_value_shape.is_dynamic() || ov::is_scalar(on_value_shape.to_shape()),
                          "on_value input must be scalar.");

    NODE_VALIDATION_CHECK(op,
                          off_value_shape.is_dynamic() || ov::is_scalar(off_value_shape.to_shape()),
                          "off_value input must be scalar.");

    auto output_shapes = std::vector<TRShape>(1);
    auto& result_shape = output_shapes[0];
    if (indices_shape.rank().is_static()) {
        result_shape = indices_shape;
        const auto axis = ov::util::try_normalize_axis(op->get_axis(), indices_shape.rank() + 1, *op);

        auto depth_as_shape =
            get_input_const_data_as_shape<TRShape>(op, 1, ta, util::GetNotNegative<typename DimType::value_type>(op));

        if (depth_as_shape && depth_as_shape->size() == 1) {
            result_shape.insert(result_shape.begin() + axis, (*depth_as_shape)[0]);
        } else {
            result_shape.insert(result_shape.begin() + axis, DimType());
        }
    } else {
        result_shape = PartialShape::dynamic();
    }
    return output_shapes;
}
}  // namespace v1
}  // namespace op
}  // namespace ov
