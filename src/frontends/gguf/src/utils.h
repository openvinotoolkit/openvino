// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <utility>

#include "node_context.h"

namespace ov {
namespace frontend {
namespace gguf {

void num_inputs_check(const NodeContext& context, size_t min_inputs, size_t max_inputs);

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::op::v3::ShapeOf>& shape,
                                         const std::vector<int>& dims);
std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node>& node, const std::vector<int>& dims);

OutputVector rename_outputs_with_suffix(const OutputVector& outputs, const std::string& suffix);

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(const RopeConfig& rope_config,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight = nullptr,
                                                           bool imrope = false);

ov::Output<ov::Node> process_view_input(const NodeContext& context, int input_index, int slice_len = 0);

namespace op {
template <typename T>
OutputVector translate_1to1_match_2_inputs(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto res = std::make_shared<T>(context.get_input(0), context.get_input(1));
    return rename_outputs_with_suffix({res}, context.get_name());
}
}  // namespace op

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
