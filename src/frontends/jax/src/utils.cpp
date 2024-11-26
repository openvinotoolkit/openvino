// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "jax_framework_node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert_promote_types.hpp"
#include "openvino/opsets/opset10.hpp"

namespace ov {
namespace frontend {
namespace jax {

void num_inputs_check(const NodeContext& context, size_t min_inputs, size_t max_inputs) {
    auto inputs = context.inputs();
    FRONT_END_OP_CONVERSION_CHECK(inputs.size() >= min_inputs, "Got less inputs than expected");
    FRONT_END_OP_CONVERSION_CHECK(inputs.size() <= max_inputs, "Got more inputs than expected");
}

void num_inputs_check(const NodeContext& context, size_t min_inputs) {
    auto inputs = context.inputs();
    FRONT_END_OP_CONVERSION_CHECK(inputs.size() >= min_inputs, "Got less inputs than expected");
}

const std::string& get_jax_prefix() {
    return jax_prefix;
}

bool is_python_scalar_input(const NodeContext& context, size_t index) {
    return context.get_input_type(index).is<type::PyScalar>();
}

void add_exception_to_fw_node(std::shared_ptr<Node> node, const std::string& msg) {
    if (auto fw_node = ov::as_type_ptr<JaxFrameworkNode>(node)) {
        auto attrs = fw_node->get_attrs();
        attrs[JaxFrameworkNode::failed_conversion_key] = msg;
        fw_node->set_attrs(attrs);
    }
}

namespace {
/*
 * Please change the corresponding map in `src/bindings/python/src/openvino/frontend/jax/utils.py`
 * if you want to change this map.
 */
const std::unordered_map<int64_t, element::Type> JAX_TO_OV_TYPE{
    {0, element::u8},
    {1, element::i8},
    {2, element::i16},
    {3, element::i32},
    {4, element::i64},
    {5, element::f16},
    {6, element::f32},
    {7, element::f64},
    {8, element::u16},
    {9, element::u32},
    {10, element::u64},
    {11, element::boolean},
    {12, element::i8},   // quantized i8
    {13, element::u8},   // quantized u8
    {14, element::i32},  // quantized i32
    {15, element::bf16},
};

std::shared_ptr<JaxFrameworkNode> create_fw_node_with_exception(const NodeContext& context,
                                                                const ov::OutputVector& inputs,
                                                                size_t num_outputs,
                                                                const std::string& exception_message,
                                                                bool skip_subgraphs = false) {
    auto fw_node = std::make_shared<JaxFrameworkNode>(context.get_decoder(), inputs, num_outputs);
    auto attrs = fw_node->get_attrs();
    std::string message(exception_message);
    if (!message.empty()) {
        message = "Exception happened during conversion of operation " + fw_node->get_friendly_name() + '\n' + message;
    }
    attrs[JaxFrameworkNode::failed_conversion_key] = message;
    fw_node->set_attrs(attrs);
    return fw_node;
}

}  // namespace

element::Type convert_dtype(int64_t jax_type) {
    FRONT_END_OP_CONVERSION_CHECK(JAX_TO_OV_TYPE.count(jax_type), "Unknown type: ", jax_type);
    return JAX_TO_OV_TYPE.at(jax_type);
};

OutputVector make_framework_node(const NodeContext& context, const std::string& exception) {
    // We create additional output for such nodes. It contains new tensor that represents input that was changed.
    auto fw_node = create_fw_node_with_exception(context, context.inputs(), context.get_output_size() + 1, exception);
    auto outputs = fw_node->outputs();
    return outputs;
}

}  // namespace jax
}  // namespace frontend
}  // namespace ov
