// Copyright (C) 2018-2022 Intel Corporation
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
    OPENVINO_UNREACHABLE("This code should be executed only for PartialShape class");
}

template <>
inline void dynamic_inference<ov::PartialShape>(const ov::PartialShape& input_shape,
                                                ov::PartialShape& output_shape,
                                                bool keep_dims) {
    output_shape = keep_dims ? ov::PartialShape::dynamic(input_shape.rank()) : ov::PartialShape::dynamic();
}

template <class T>
void reduce_shape_infer(const ov::op::util::ReductionBase* op,
                        bool keep_dims,
                        const T& input_shape,
                        T& output_shape,
                        const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    const auto& data_rank = input_shape.rank();
    std::vector<int64_t> axes_val;
    bool axes_are_known = get_data_as_int64<T>(1, op, axes_val, constant_data);

    if (data_rank.is_static() && axes_are_known) {
        ov::normalize_axes(op, data_rank.get_length(), axes_val);

        if (keep_dims) {
            output_shape = input_shape;
            for (const auto& axis : axes_val)
                output_shape[axis] = 1;
            return;
        }
        for (int64_t i = 0; i < data_rank.get_length(); ++i)
            if (find(axes_val.begin(), axes_val.end(), i) == axes_val.end())
                output_shape.push_back(input_shape[i]);
    } else {
        dynamic_inference(input_shape, output_shape, keep_dims);
    }
}

template <class T>
void shape_infer(const ov::op::util::ArithmeticReductionKeepDims* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);
    reduce_shape_infer(op, op->get_keep_dims(), input_shapes[0], output_shapes[0], constant_data);
}

template <class T>
void shape_infer(const ov::op::util::LogicalReductionKeepDims* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);
    reduce_shape_infer(op, op->get_keep_dims(), input_shapes[0], output_shapes[0], constant_data);
}
