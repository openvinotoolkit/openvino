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
    for (auto i = max_inputs; i < inputs.size(); i++) {
        FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(i), "Got more inputs than expected.");
    }
}

const std::string& get_jax_prefix() {
    return jax_prefix;
}

bool is_none_node(const Output<Node>& node) {
    if (const auto& fw_node_inp = std::dynamic_pointer_cast<ov::op::util::FrameworkNode>(node.get_node_shared_ptr())) {
        const auto& attrs = fw_node_inp->get_attrs();
        if (attrs.find("none_value") != attrs.end()) {
            return true;
        }
    }
    return false;
}

Any simplified_type_interpret(Any type) {
    // Type in jaxpr is already the dtype.
    return type;
}

bool is_python_scalar_input(const NodeContext& context, size_t index) {
    return context.get_input_type(index).is<type::PyScalar>();
}

void align_eltwise_input_types(const NodeContext& context,
                               Output<Node>& lhs,
                               Output<Node>& rhs,
                               const bool& is_lhs_python_scalar,
                               const bool& is_rhs_python_scalar) {
    const auto& lhs_type = lhs.get_element_type();
    const auto& rhs_type = rhs.get_element_type();
    auto const_0 = ov::op::v0::Constant::create(element::i32, Shape{}, {1});
    auto const_1 = ov::op::v0::Constant::create(element::i32, Shape{1}, {1});
    // Create temporary copy of lhs and rhs for ConvertPromoteTypes to not modify original nodes.
    ov::Output<ov::Node> tmp_lhs = lhs;
    ov::Output<ov::Node> tmp_rhs = rhs;
    // Python scalar has lower priority than any tensor with any dimension.
    // If only one input is PyScalar, replace it with const to mitigate issues with dynamic type caused by dynamic
    // shape.
    if (is_lhs_python_scalar && !is_rhs_python_scalar) {
        tmp_lhs = context.mark_node(std::make_shared<opset10::ConvertLike>(const_0, lhs));
        tmp_rhs = context.mark_node(std::make_shared<opset10::ConvertLike>(const_1, rhs));
    } else if (!is_lhs_python_scalar && is_rhs_python_scalar) {
        tmp_lhs = context.mark_node(std::make_shared<opset10::ConvertLike>(const_1, lhs));
        tmp_rhs = context.mark_node(std::make_shared<opset10::ConvertLike>(const_0, rhs));
    }

    auto at = context.mark_node(
        std::make_shared<ov::op::v14::ConvertPromoteTypes>(tmp_lhs, tmp_rhs, true, true, element::f32));
    auto dst_type = at->get_output_element_type(0);
    if (dst_type.is_dynamic()) {
        // Add ConvertLike on original node to not remove changes to shape done to differentiate between tensors and
        // scalars.
        lhs = context.mark_node(std::make_shared<opset10::ConvertLike>(lhs, at->output(0)));
        rhs = context.mark_node(std::make_shared<opset10::ConvertLike>(rhs, at->output(1)));
    } else {
        // Cast to destination type
        if (dst_type != lhs_type) {
            lhs = context.mark_node(std::make_shared<opset10::Convert>(lhs, dst_type));
        }
        if (dst_type != rhs_type) {
            rhs = context.mark_node(std::make_shared<opset10::Convert>(rhs, dst_type));
        }
    }
    return;
}

void align_output_types(const NodeContext& context, OutputVector& outputs) {
    for (size_t i = 0; i < outputs.size(); i++) {
        auto dtype_any = context.get_output_type(i);
        if (dtype_any.is<element::Type>()) {
            auto dtype = dtype_any.as<element::Type>();
            if (dtype.is_static() && dtype != outputs[i].get_element_type()) {
                outputs[i] = std::make_shared<opset10::Convert>(outputs[i], dtype);
            }
        }
    }
}

Output<Node> get_input_with_floating_type(const NodeContext& context, size_t idx) {
    auto x = context.get_input(static_cast<int>(idx));
    // This const only needed for type alignment
    auto dummy_const = context.mark_node(ov::op::v0::Constant::create(element::f32, Shape({}), {0.5}))->output(0);
    align_eltwise_input_types(context, x, dummy_const, false, true);
    return x;
}

Output<Node> get_input_as_i32(const NodeContext& context, size_t idx) {
    auto x = context.get_input(static_cast<int>(idx));
    if (x.get_element_type() != element::i32) {
        x = context.mark_node(std::make_shared<ov::op::v0::Convert>(x, element::i32));
    }
    return x;
}

std::tuple<Output<Node>, Output<Node>> get_inputs_with_promoted_types(const NodeContext& context,
                                                                      size_t lhs_idx,
                                                                      size_t rhs_idx) {
    FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(lhs_idx) && !context.input_is_none(rhs_idx),
                                  "Input should not be None.");
    auto lhs = context.get_input(static_cast<int>(lhs_idx));
    auto rhs = context.get_input(static_cast<int>(rhs_idx));
    align_eltwise_input_types(context,
                              lhs,
                              rhs,
                              is_python_scalar_input(context, lhs_idx),
                              is_python_scalar_input(context, rhs_idx));
    return std::make_tuple(lhs, rhs);
}

void add_exception_to_fw_node(std::shared_ptr<Node> node, const std::string& msg) {
    if (auto fw_node = ov::as_type_ptr<JaxFrameworkNode>(node)) {
        auto attrs = fw_node->get_attrs();
        attrs[JaxFrameworkNode::failed_conversion_key] = msg;
        fw_node->set_attrs(attrs);
    }
}

void copy_runtime_info_and_name(const std::shared_ptr<Node>& from,
                                ov::NodeVector to,
                                const ov::NodeVector& additional_rt_info_src) {
    if (to.size() == 1) {
        // We do 1 to 1 matching, no need to process names, just inherit initial name
        to[0]->set_friendly_name(from->get_friendly_name());
    } else {
        std::unordered_set<std::string> unique_names;
        size_t idx = 0;
        for (auto& op : to) {
            auto new_name = from->get_friendly_name() + '/' + op->get_type_name();
            if (unique_names.count(new_name)) {
                new_name += '_' + std::to_string(idx++);
            } else {
                unique_names.insert(new_name);
            }
            op->set_friendly_name(new_name);
        }
    }
    copy_runtime_info(from, to);
    if (!additional_rt_info_src.empty())
        copy_runtime_info(additional_rt_info_src, to);
}

// helper ops
Output<Node> masked_fill(ov::pass::NodeRegistry& rg,
                         const Output<Node>& data,
                         const Output<Node>& mask,
                         const Output<Node>& value) {
    auto _value = rg.make<opset10::ConvertLike>(value, data);
    auto bool_mask = rg.make<opset10::Convert>(mask, element::boolean);
    return rg.make<opset10::Select>(bool_mask, _value, data);
}

}  // namespace jax
}  // namespace frontend
}  // namespace ov
