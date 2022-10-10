// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/core/validation_util.hpp>
#include <openvino/op/util/arithmetic_reductions_keep_dims.hpp>
#include <openvino/op/util/logical_reduction_keep_dims.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset3.hpp>

#include "utils.hpp"

template <class T>
void shape_infer(const ov::opset1::Reshape* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);
    std::vector<int64_t> output_pattern;
    bool status = get_data_as_int64<T>(1, op, output_pattern, constant_data);
    NODE_VALIDATION_CHECK(op, status, "Shape inference lacks input data");

    auto& input_shape = input_shapes[0];
    OPENVINO_ASSERT(input_shape.is_static());
    auto& output_shape = output_shapes[0];
    output_shape.resize(output_pattern.size());

    auto output_rank = input_shapes[1].size() == 0 ? 0 : input_shapes[1][0];
    if (output_rank == 0 && output_shape.size() != 0) {
        output_pattern.clear();
        OPENVINO_ASSERT(output_pattern.size() == 1);
        NODE_VALIDATION_CHECK(op, output_pattern[0] == 1, "The value of scalar shape pattern should be equal to 1!");
    }

    auto special_zero = op->get_special_zero();

    size_t output_product(1);
    int64_t minus_one_idx = -1;
    for (size_t i = 0; i < output_pattern.size(); ++i) {
        if (output_pattern[i] == -1) {  // resolving everything except -1
            NODE_VALIDATION_CHECK(op,
                                  minus_one_idx == -1,
                                  "More than one element of output shape pattern has value of -1");
            minus_one_idx = static_cast<int64_t>(i);
            continue;
        }

        auto pattern_dim = output_pattern[i];
        if (pattern_dim == 0 && special_zero) {
            NODE_VALIDATION_CHECK(op, i < input_shape.size(), "'0' dimension is out of range");
            output_shape[i] = input_shape[i];
            // we do not include dimension to output product here and won't include in input
            // product later because we will divide output_product by input_product. This
            // dimension contributes to both products equally
        } else {
            output_shape[i] = pattern_dim;
            output_product *= pattern_dim;
        }
    }
    size_t input_product(1);
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (i < output_pattern.size() && output_pattern[i] == 0)
            continue;
        input_product = input_shape[i].get_length() * input_product;
    }

    if (minus_one_idx != -1)  // resolving -1 masked dimension
    {
        if (output_product == 0) {
            NODE_VALIDATION_CHECK(op,
                                  input_product == 0,
                                  "Cannot infer '-1' dimension with zero-size output "
                                  "dimension unless at least one input dimension is "
                                  "also zero-size");
            output_shape[minus_one_idx] = 0;
        } else {
            NODE_VALIDATION_CHECK(op,
                                  input_product % output_product == 0,
                                  "Non-'-1' output dimensions do not evenly divide the input dimensions");
            output_shape[minus_one_idx] = input_product / output_product;
        }
    }

    size_t zero_dims = std::count_if(output_pattern.begin(), output_pattern.end(), [](const int64_t& dim) {
        return dim == 0;
    });

    bool backward_compatible_check = (zero_dims && special_zero) || minus_one_idx != -1;
    bool in_out_elements_equal = input_product == output_product;

    NODE_VALIDATION_CHECK(op,
                          backward_compatible_check || in_out_elements_equal,
                          "Requested output shape ",
                          output_shape,
                          " is incompatible with input shape ",
                          input_shape);
}

template <class T>
void shape_infer(const ov::opset1::Squeeze* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);
    std::vector<int64_t> axes;
    bool axes_is_constant = get_data_as_int64<T>(1, op, axes, constant_data);
    NODE_VALIDATION_CHECK(op, axes_is_constant, "Shape inference lacks input data");

    auto& input_shape = input_shapes[0];
    OPENVINO_ASSERT(input_shape.is_static());
    auto& output_shape = output_shapes[0];
    output_shape = T{};

    ov::normalize_axes(op, input_shape.rank().get_length(), axes);

    for (uint64_t idx = 0; idx < input_shape.size(); ++idx) {
        if (std::find(axes.begin(), axes.end(), idx) == axes.end()) {
            output_shape.push_back(input_shape[idx]);
        } else {
            NODE_VALIDATION_CHECK(op,
                                  input_shape[idx] == 1,
                                  "provided axis value is invalid. Only axes of size 1 may be removed.");
        }
    }
}

template <class T>
void shape_infer(const ov::opset1::Unsqueeze* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);
    std::vector<int64_t> axes;
    bool axes_is_constant = get_data_as_int64<T>(1, op, axes, constant_data);
    NODE_VALIDATION_CHECK(op, axes_is_constant, "Shape inference lacks input data");

    auto& input_shape = input_shapes[0];
    OPENVINO_ASSERT(input_shape.is_static());
    auto& output_shape = output_shapes[0];
    output_shape = input_shape;

    NODE_VALIDATION_CHECK(op, !axes.empty(), "'axes' input is mandatory");

    int64_t expanded_rank = input_shape.size() + axes.size();
    ov::normalize_axes(op, static_cast<int64_t>(expanded_rank), axes);

    std::set<int64_t> unique_sorted_axes(axes.begin(), axes.end());
    for (const auto& axis : unique_sorted_axes) {
        NODE_VALIDATION_CHECK(op, axis <= expanded_rank, "provided 'axes' value ", axis, " is not valid.");
        output_shape.insert(next(output_shape.begin(), axis), 1);
    }
}

template <class T>
inline void dynamic_shape(T& output_shape) {
    OPENVINO_UNREACHABLE("This code should be executed only for PartialShape class");
}

template <>
inline void dynamic_shape<ov::PartialShape>(ov::PartialShape& output_shape) {
    output_shape = ov::PartialShape::dynamic();
}

template <class T>
void shape_of_shape_infer(const T& input_shape, T& output_shape) {
    if (input_shape.rank().is_static()) {
        const auto& rank = input_shape.size();
        if (rank) {
            output_shape.resize(1);
            output_shape[0] = rank;
        } else {
            output_shape.clear();
        }
    } else {
        dynamic_shape(output_shape);
    }
}

template <class T>
void shape_infer(const ov::opset1::ShapeOf* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1 && output_shapes.size() == 1);
    shape_of_shape_infer(input_shapes[0], output_shapes[0]);
}

template <class T>
void shape_infer(const ov::opset3::ShapeOf* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1 && output_shapes.size() == 1);
    shape_of_shape_infer(input_shapes[0], output_shapes[0]);
}
