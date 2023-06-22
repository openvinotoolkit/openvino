// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/opsets/opset12.hpp"

void shift_left(std::vector<size_t>& vec, size_t k);

void shift_right(std::vector<size_t>& vec, size_t k);

std::vector<size_t> gather_forward(size_t size, size_t initial_value);

std::vector<size_t> gather_backward(size_t size, size_t initial_value);

template <typename CreateIndicesF>
std::shared_ptr<ov::opset12::Gather> make_gather(std::shared_ptr<ov::Node> input_node,
                                                 CreateIndicesF create_indices_func,
                                                 size_t axis) {
    const ov::Shape& input_shape = input_node->get_output_shape(0);
    const std::vector<size_t> indexes = create_indices_func(input_shape[axis], 0);

    auto gather_indexes_node = ov::opset12::Constant::create(ov::element::i64, ov::Shape{indexes.size()}, indexes);

    auto gather_axis_node = ov::opset12::Constant::create(ov::element::i64, ov::Shape{}, {axis});

    return std::make_shared<ov::opset12::Gather>(input_node->output(0), gather_indexes_node, gather_axis_node);
}
