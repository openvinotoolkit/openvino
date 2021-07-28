// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/util/arithmetic_reductions_keep_dims.hpp>
#include <openvino/op/util/logical_reduction_keep_dims.hpp>
#include <openvino/core/validation_util.hpp>

template<class T>
void reduce_shape_infer(const ov::op::util::ReductionBase* op, bool keep_dims, const T& input_shape, T& output_shape) {
    const auto& data_rank = input_shape.rank();
    const auto& axes = ov::get_constant_from_source(op->input_value(1));

    if (data_rank.is_static() && axes) {
        auto axes_val = axes->cast_vector<int64_t>();
        ov::normalize_axes(op, data_rank.get_length(), axes_val);

        if (keep_dims)
        {
            output_shape = input_shape;
            for (const auto& axis : axes_val)
                output_shape[axis] = 1;
            return;
        }
        for (int64_t i = 0; i < data_rank.get_length(); ++i)
            if (find(axes_val.begin(), axes_val.end(), i) == axes_val.end())
                output_shape.push_back(input_shape[i]);
    } else if (data_rank.is_static() && keep_dims) {
        output_shape.resize(data_rank.get_length());
    } // otherwise it should stay dynamic as it came to the function
}

template<class T>
void shape_infer(const ov::op::util::ArithmeticReductionKeepDims* op, const std::vector<T> &input_shapes, std::vector<T> &output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);
    reduce_shape_infer(op, op->get_keep_dims(), input_shapes[0], output_shapes[0]);
}

template<class T>
void shape_infer(const ov::op::util::LogicalReductionKeepDims* op, const std::vector<T> &input_shapes, std::vector<T> &output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);
    reduce_shape_infer(op, op->get_keep_dims(), input_shapes[0], output_shapes[0]);
}
