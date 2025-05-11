// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/arithmetic_reductions_keep_dims.hpp"
#include "openvino/op/util/logical_reduction_keep_dims.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace util {

template <class T>
result_shape_t<T> reduce_shape(const T& input_shape, std::vector<int64_t>& axes, const bool keep_dims) {
    if (keep_dims) {
        result_shape_t<T> result = input_shape;
        result = input_shape;
        for (auto&& axis : axes) {
            result[axis] = 1;
        }
        return result;
    } else {
        const auto s = input_shape.size();
        result_shape_t<T> result;
        result.reserve(s);

        for (size_t axis = 0; axis < s; ++axis) {
            if (std::find(axes.begin(), axes.end(), axis) == axes.end()) {
                result.emplace_back(input_shape[axis]);
            }
        }
        return result;
    }
}
}  // namespace util

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> reduce_shape_infer(const util::ReductionBase* op,
                                        bool keep_dims,
                                        const std::vector<TShape>& input_shapes,
                                        const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);

    const auto& data_shape = input_shapes[0];
    const auto& data_rank = data_shape.rank();
    const auto& axes_shape = input_shapes[1];
    const auto& axes_rank = axes_shape.rank();

    std::vector<TRShape> output_shapes;
    output_shapes.reserve(1);

    NODE_VALIDATION_CHECK(op,
                          axes_rank.compatible(0) || axes_rank.compatible(1),
                          "Axes input must be a scalar or 1D input. Got: ",
                          axes_shape);

    auto axes_val = ov::op::get_input_const_data_as<TRShape, int64_t>(op, 1, tensor_accessor);

    if (data_rank.is_static() && axes_val) {
        ov::util::try_normalize_axes(*axes_val, data_rank, *op);

        output_shapes.push_back(util::reduce_shape(data_shape, *axes_val, keep_dims));
    } else {
        if (keep_dims) {
            output_shapes.push_back(ov::PartialShape::dynamic(data_shape.rank()));
        } else {
            if (axes_shape.is_static() && shape_size(axes_shape.to_shape()) == 1) {
                // since there is the only axis, it is unique/not duplicated by definition. we can safely propagate rank
                output_shapes.push_back(ov::PartialShape::dynamic(data_rank - 1));
            } else {
                output_shapes.push_back(ov::PartialShape::dynamic());
            }
        }
    }
    return output_shapes;
}

// API: TensorAccessor to constant data
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const util::ArithmeticReductionKeepDims* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    return reduce_shape_infer(op, op->get_keep_dims(), input_shapes, tensor_accessor);
}

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const util::LogicalReductionKeepDims* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    return reduce_shape_infer(op, op->get_keep_dims(), input_shapes, tensor_accessor);
}
}  // namespace op
}  // namespace ov
