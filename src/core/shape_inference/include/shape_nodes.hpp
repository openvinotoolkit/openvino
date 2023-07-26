// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/core/validation_util.hpp>
#include <openvino/op/util/arithmetic_reductions_keep_dims.hpp>
#include <openvino/op/util/logical_reduction_keep_dims.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset3.hpp>

#include "utils.hpp"

template <class T, class TRShape = ov::result_shape_t<T>>
std::vector<TRShape> shape_infer(const ov::op::v1::Reshape* op,
                                 const std::vector<T>& input_shapes,
                                 const ov::ITensorAccessor& ta = ov::make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);
    auto output_pattern = ov::op::get_input_const_data_as<TRShape, int64_t>(op, 1, ta);
    NODE_VALIDATION_CHECK(op, output_pattern, "Shape inference lacks input data");

    auto& input_shape = input_shapes[0];
    OPENVINO_ASSERT(input_shape.is_static());
    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];
    output_shape.resize(output_pattern->size());

    auto output_rank = input_shapes[1].size() == 0 ? 0 : input_shapes[1][0];
    if (output_rank == 0 && output_shape.size() != 0) {
        output_pattern->clear();
        OPENVINO_ASSERT(output_pattern->size() == 1);
        NODE_VALIDATION_CHECK(op, (*output_pattern)[0] == 1, "The value of scalar shape pattern should be equal to 1!");
    }

    auto special_zero = op->get_special_zero();

    size_t output_product(1);
    int64_t minus_one_idx = -1;
    for (size_t i = 0; i < output_pattern->size(); ++i) {
        if ((*output_pattern)[i] == -1) {  // resolving everything except -1
            NODE_VALIDATION_CHECK(op,
                                  minus_one_idx == -1,
                                  "More than one element of output shape pattern has value of -1");
            minus_one_idx = static_cast<int64_t>(i);
            continue;
        }

        auto pattern_dim = (*output_pattern)[i];
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
        if (i < output_pattern->size() && (*output_pattern)[i] == 0 && special_zero)
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

    size_t zero_dims = std::count_if(output_pattern->begin(), output_pattern->end(), [](const int64_t& dim) {
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

    return output_shapes;
}

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
