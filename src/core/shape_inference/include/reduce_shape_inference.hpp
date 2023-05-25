// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/core/validation_util.hpp>
#include <openvino/op/util/arithmetic_reductions_keep_dims.hpp>
#include <openvino/op/util/logical_reduction_keep_dims.hpp>
#include <openvino/opsets/opset1.hpp>

#include "utils.hpp"

template <class TShape>
std::vector<TShape> reduce_shape_infer(const ov::op::util::ReductionBase* op,
                                       bool keep_dims,
                                       const std::vector<TShape>& input_shapes,
                                       const ov::ITensorAccessor& tensor_accessor = ov::make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);

    const auto& data_shape = input_shapes[0];
    const auto& data_rank = data_shape.rank();
    const auto& axes_shape = input_shapes[1];
    const auto& axes_rank = axes_shape.rank();

    std::vector<TShape> output_shapes;
    output_shapes.reserve(1);

    NODE_VALIDATION_CHECK(op,
                          axes_rank.compatible(0) || axes_rank.compatible(1),
                          "Axes input must be a scalar or 1D input. Got: ",
                          axes_shape);

    const auto axes_val = ov::op::get_input_const_data_as<TShape, int64_t>(op, 1, tensor_accessor);

    if (data_rank.is_static() && axes_val) {
        OPENVINO_SUPPRESS_DEPRECATED_START
        ov::normalize_axes(op, data_rank.get_length(), *axes_val);
        OPENVINO_SUPPRESS_DEPRECATED_END

        if (keep_dims) {
            output_shapes.push_back(data_shape);
            TShape& output_shape = output_shapes[0];
            for (const auto& axis : *axes_val) {
                output_shape[axis] = 1;
            }
        } else {
            output_shapes.resize(1);
            TShape& output_shape = output_shapes[0];
            for (size_t i = 0; i < data_shape.size(); ++i) {
                if (std::find(axes_val->begin(), axes_val->end(), i) == axes_val->end()) {
                    output_shape.push_back(data_shape[i]);
                }
            }
        }
    } else {
        if (keep_dims) {
            output_shapes.push_back(ov::PartialShape::dynamic(data_shape.rank()));
        } else {
            output_shapes.push_back(ov::PartialShape::dynamic());
        }
    }
    return output_shapes;
}

// API: TensorAccessor to constant data
template <class TShape>
std::vector<TShape> shape_infer(const ov::op::util::ArithmeticReductionKeepDims* op,
                                const std::vector<TShape>& input_shapes,
                                const ov::ITensorAccessor& tensor_accessor = ov::make_tensor_accessor()) {
    return reduce_shape_infer(op, op->get_keep_dims(), input_shapes, tensor_accessor);
}

template <class TShape>
std::vector<TShape> shape_infer(const ov::op::util::LogicalReductionKeepDims* op,
                                const std::vector<TShape>& input_shapes,
                                const ov::ITensorAccessor& tensor_accessor = ov::make_tensor_accessor()) {
    return reduce_shape_infer(op, op->get_keep_dims(), input_shapes, tensor_accessor);
}

// API for compatibility: Constant data map
template <class TShape>
void shape_infer(const ov::op::util::ArithmeticReductionKeepDims* op,
                 const std::vector<TShape>& input_shapes,
                 std::vector<TShape>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    const auto tensor_accessor = ov::make_tensor_accessor(constant_data);
    output_shapes = reduce_shape_infer(op, op->get_keep_dims(), input_shapes, tensor_accessor);
}

template <class TShape>
void shape_infer(const ov::op::util::LogicalReductionKeepDims* op,
                 const std::vector<TShape>& input_shapes,
                 std::vector<TShape>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    const auto tensor_accessor = ov::make_tensor_accessor(constant_data);
    output_shapes = reduce_shape_infer(op, op->get_keep_dims(), input_shapes, tensor_accessor);
}
