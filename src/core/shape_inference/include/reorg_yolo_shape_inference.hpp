// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/reorg_yolo.hpp>

#include "utils.hpp"
namespace ov {
namespace op {
namespace v0 {

template <class T>
void shape_infer(const ReorgYolo* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 1) && output_shapes.size() == 1);
    const auto& input_shape = input_shapes[0];
    auto& output_shape = output_shapes[0];
    auto strides = op->get_strides();
    if (input_shape.is_static()) {
        NODE_VALIDATION_CHECK(op, input_shape.size() == 4, "[N, C, H, W] input shape is required.");

        NODE_VALIDATION_CHECK(op,
                              (input_shape[2].get_length() % strides[0]) == 0,
                              "For [N, C, H, W] input shape, H should be divisible by stride.");

        NODE_VALIDATION_CHECK(op,
                              (input_shape[3].get_length() % strides[0]) == 0,
                              "For [N, C, H, W] input shape, W should be divisible by stride.");

        NODE_VALIDATION_CHECK(op,
                              input_shape[1].get_length() >= (strides[0] * strides[0]),
                              "For [N, C, H, W] input shape, C >= (stride*stride) is required.");

        output_shape = T({input_shape[0], input_shape[1]});

        for (size_t i = 2; i < input_shape.size(); i++) {
            output_shape.push_back(input_shape[i].get_length() / strides[0]);
            output_shape[1] *= strides[0];
        }
    } else {
        output_shape = ov::PartialShape::dynamic(input_shape.rank());
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov
