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
}

const std::string& get_jax_prefix() {
    return jax_prefix;
}

Any simplified_type_interpret(Any type) {
    // Type in jaxpr is already the dtype.
    return type;
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
const std::unordered_map<int64_t, element::Type> JAX_TO_OV_TYPE{
    {0, element::u8},
    {1, element::i8},
    {2, element::i16},
    {3, element::i32},
    {4, element::i64},
    {5, element::f16},
    {6, element::f32},
    {7, element::f64},
    {11, element::boolean},
    {12, element::i8},   // quantized i8
    {13, element::u8},   // quantized u8
    {14, element::i32},  // quantized i32
    {15, element::bf16},
};
}  // namespace

element::Type convert_dtype(int64_t pt_type) {
    FRONT_END_OP_CONVERSION_CHECK(JAX_TO_OV_TYPE.count(pt_type), "Unknown type: ", pt_type);
    return JAX_TO_OV_TYPE.at(pt_type);
};

}  // namespace jax
}  // namespace frontend
}  // namespace ov
