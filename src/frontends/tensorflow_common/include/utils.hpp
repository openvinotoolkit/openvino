// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/graph_rewrite.hpp"

#ifndef TENSORFLOW_OP_VALIDATION
#    define TENSORFLOW_OP_VALIDATION(node_context, ...)                                        \
        OPENVINO_ASSERT_HELPER(::ov::frontend::OpValidationFailure,                            \
                               ("While validating node '" + node_context.get_op_type() + "'"), \
                               __VA_ARGS__)
#endif

namespace ov {
namespace frontend {
namespace tensorflow {
using OpMap = std::unordered_map<std::string, std::vector<ov::Output<ov::Node>>>;

void extract_operation_name_and_port(const std::string& port_name,
                                     std::string& operation_name,
                                     size_t& port_index,
                                     std::string& port_type);

void set_out_name(const std::string& out_name, const Output<Node>& output);

void set_node_name(const std::string& node_name, const std::shared_ptr<Node>& node);

bool is_conditional_edge(const std::string& input_tensor_name);

template <typename T>
void get_const_input(const NodeContext& node, int input_index, std::vector<T>* vector) {
    auto input_size = static_cast<int>(node.get_input_size());
    auto node_name = node.get_name();
    auto node_type = node.get_op_type();
    FRONT_END_GENERAL_CHECK(0 <= input_index && input_index < input_size,
                            "[TensorFlow Frontend] Internal error: Node " + node_name + " has " +
                                std::to_string(input_size) + " inputs, but requested input port index to be " +
                                std::to_string(input_size));
    auto ov_input = node.get_input(input_index);
    if (auto constant = get_constant_from_source(ov_input)) {
        *vector = constant->cast_vector<T>();
        return;
    }
    FRONT_END_THROW("[TensorFlow Frontend] Internal error: Input " + std::to_string(input_index) +
                    " cannot be folded to Constant for node " + node_name + " of type " + node_type);
}

ov::op::PadType convert_tf_padding(const NodeContext& node, const std::string& tf_padding);

ov::OutputVector translate_convolution_op(const NodeContext& node, size_t spatial_dims_num);

void fill_explicit_pads_vectors(const NodeContext& node,
                                bool is_nhwc,
                                size_t spatial_dims_num,
                                const std::vector<int64_t>& tf_explicit_paddings,
                                ov::CoordinateDiff& pads_begin,
                                ov::CoordinateDiff& pads_end);

void default_op_checks(const NodeContext& node, size_t min_input_size, const std::vector<std::string>& supported_ops);

ov::Output<Node> get_elements_number_1d(const Output<Node>& output,
                                        ov::element::Type output_type,
                                        ov::pass::NodeRegistry& rg);

ov::op::PadMode convert_padding_mode(const NodeContext& node, const std::string& padding_mode);

Output<Node> compute_subgraph_scalar_rank(const Output<Node>& output,
                                          element::Type output_type,
                                          bool as_scalar = false);
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
