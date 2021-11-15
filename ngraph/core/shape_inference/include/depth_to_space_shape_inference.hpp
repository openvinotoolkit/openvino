// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <openvino/core/validation_util.hpp>
#include <openvino/op/depth_to_space.hpp>
#include <openvino/opsets/opset1.hpp>

#include "utils.hpp"
namespace ov {
namespace op {
namespace v0 {

template <class T>
void shape_infer(const ov::op::v0::DepthToSpace* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1 && output_shapes.size() == 1);

    const auto& data_shape = input_shapes[0];
    const ov::Rank data_rank = data_shape.rank();
    const auto block_size = op->get_block_size();
    size_t divider = 0;

    if (data_rank.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              !(data_shape.size() < 3),
                              "The input tensor with rank lower than 3 is not supported (input rank: ",
                              data_shape.size(),
                              ")");

        divider = std::pow(block_size, data_shape.size() - 2);
        NODE_VALIDATION_CHECK(op, (divider), "DepthToSpace: The divider must not be 0");
    }

    if (data_shape.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              block_size > 0 && !(data_shape[1].get_length() % block_size),
                              "DepthToSpace: The input data's 'channels' axis size: ",
                              data_shape[1],
                              " must be a equivalent to 'block_size'^'spatial_dims': ",
                              divider);
        auto& output_shape = output_shapes[0];

        output_shape.resize(data_shape.size());
        output_shape[0] = data_shape[0].get_length();
        output_shape[1] = data_shape[1].get_length() / divider;
        for (size_t i = 2; i < output_shape.size(); i++) {
            output_shape[i] = data_shape[i].get_length() * block_size;
        }
    } else {
        // For PartialShape, Set the output to be dynamic;
        // For StaticShape, throw error caused by implicitly constructing StaticShape with PartialShape argument;
        output_shapes[0] = ov::PartialShape::dynamic(data_rank);
    }
}

}  // namespace v0
}  // namespace op
}  // namespace ov