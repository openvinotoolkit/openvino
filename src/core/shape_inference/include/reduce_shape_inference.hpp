// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/core/validation_util.hpp>
#include <openvino/op/util/arithmetic_reductions_keep_dims.hpp>
#include <openvino/op/util/logical_reduction_keep_dims.hpp>
#include <openvino/opsets/opset1.hpp>

#include "utils.hpp"

template <class T>
inline void dynamic_inference(const T& input_shape, T& output_shape, bool keep_dims) {
    OPENVINO_THROW("This code should be executed only for PartialShape class");
}

template <>
inline void dynamic_inference<ov::PartialShape>(const ov::PartialShape& input_shape,
                                                ov::PartialShape& output_shape,
                                                bool keep_dims) {
    output_shape = keep_dims ? ov::PartialShape::dynamic(input_shape.rank()) : ov::PartialShape::dynamic();
}

template <class TShape>
std::vector<TShape> reduce_shape_infer(const ov::op::util::ReductionBase* op,
                                       bool keep_dims,
                                       const std::vector<TShape>& input_shapes,
                                       const ov::ITensorAccessor& tensor_accessor = ov::make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() >= 1);

    const auto& input_shape = input_shapes[0];
    const auto& data_rank = input_shape.rank();
    TShape output_shape;

    const auto axes_val =
        ov::op::get_input_const_data_as<TShape, int64_t>(op, 1, tensor_accessor, ov::util::Cast<int64_t>());

    if (data_rank.is_static() && axes_val) {
        OPENVINO_SUPPRESS_DEPRECATED_START
        ov::normalize_axes(op, data_rank.get_length(), *axes_val);
        OPENVINO_SUPPRESS_DEPRECATED_END

        if (keep_dims) {
            output_shape = input_shape;
            for (const auto& axis : *axes_val) {
                output_shape[axis] = 1;
            }
        } else {
            for (size_t i = 0; i < input_shape.size(); ++i) {
                if (std::find(axes_val->begin(), axes_val->end(), i) == axes_val->end()) {
                    output_shape.push_back(input_shape[i]);
                }
            }
        }
    } else {
        dynamic_inference(input_shape, output_shape, keep_dims);
    }
    return {output_shape};
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
