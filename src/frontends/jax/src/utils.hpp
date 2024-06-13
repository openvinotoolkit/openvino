// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"

namespace ov {

namespace op {
namespace util {
class FrameworkNode;
}  // namespace util
}  // namespace op

namespace frontend {
namespace jax {

const std::string jax_prefix = "[JAX Frontend] ";

const std::string& get_jax_prefix();

/// \brief Macro to check whether a boolean condition holds.
/// \param COND Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evalutes to `false`.
/// \throws ::ov::frontend::OpConversionFailure if `cond` is false.
#ifndef JAX_OP_CONVERSION_CHECK
#    define JAX_OP_CONVERSION_CHECK(COND, ...) \
        OPENVINO_ASSERT_HELPER(::ov::frontend::OpConversionFailure, "", (COND), get_jax_prefix(), __VA_ARGS__)
#endif

void num_inputs_check(const NodeContext& context, size_t min_inputs, size_t max_inputs);

bool is_none_node(const Output<Node>& node);

// TODO: Eliminate the need of this function by implementing more accurate custom data type handling
Any simplified_type_interpret(Any type);

void add_exception_to_fw_node(std::shared_ptr<Node> node, const std::string& msg);

bool is_python_scalar_input(const NodeContext& context, size_t index);

void align_eltwise_input_types(const NodeContext& context,
                               Output<Node>& lhs,
                               Output<Node>& rhs,
                               const bool& is_lhs_python_scalar = false,
                               const bool& ir_rhs_python_scalar = false);
void align_output_types(const NodeContext& context, OutputVector& outputs);

Output<Node> get_input_with_floating_type(const NodeContext& context, size_t idx);

Output<Node> get_input_as_i32(const NodeContext& context, size_t idx);

std::tuple<Output<Node>, Output<Node>> get_inputs_with_promoted_types(const NodeContext& context,
                                                                      size_t lhs_idx,
                                                                      size_t rhs_idx);

namespace op {
template <OutputVector (*T)(const NodeContext&), size_t idx = 0>
OutputVector inplace_op(const NodeContext& context) {
    auto translation_res = T(context);
    FRONT_END_OP_CONVERSION_CHECK(translation_res.size() == 1,
                                  "inplace_op function must be used on single output translators");
    return translation_res;
}

template <OutputVector (*T)(const NodeContext&), size_t idx>
OutputVector optional_out(const NodeContext& context) {
    auto translation_res = T(context);
    return translation_res;
}

template <typename T>
OutputVector translate_1to1_match_1_inputs(const NodeContext& context) {
    auto res = context.mark_node(std::make_shared<T>(context.get_input(0)));
    auto out_type = context.get_output_type(0);
    if (out_type.is<element::Type>()) {
        auto dtype = out_type.as<element::Type>();
        if (dtype.is_static() && dtype != res->output(0).get_element_type()) {
            res = context.mark_node(std::make_shared<ov::op::v0::Convert>(res, dtype));
        }
    }
    return {res};
}

template <typename T>
OutputVector translate_1to1_match_1_inputs_with_fp32_type_alignment(const NodeContext& context) {
    auto x = get_input_with_floating_type(context, 0);
    return {context.mark_node(std::make_shared<T>(x))};
}

template <typename T>
OutputVector translate_1to1_match_2_inputs(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    return {context.mark_node(std::make_shared<T>(context.get_input(0), context.get_input(1)))};
}

template <typename T>
OutputVector translate_1to1_match_2_inputs_align_types(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);
    auto lhs_type = context.get_input_type(0);
    auto rhs_type = context.get_input_type(1);
    // If type is string or None, we shouldn't align
    if (!lhs_type.is<type::Str>() && !rhs_type.is<type::Str>() && !lhs_type.is<type::PyNone>() &&
        !rhs_type.is<type::PyNone>()) {
        align_eltwise_input_types(context,
                                  lhs,
                                  rhs,
                                  is_python_scalar_input(context, 0),
                                  is_python_scalar_input(context, 1));
    }
    OutputVector res = {context.mark_node(std::make_shared<T>(lhs, rhs))};
    align_output_types(context, res);
    return res;
}

template <typename T, size_t idx = 0>
OutputVector inplace_translate_1to1_match_2_inputs_align_types(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);
    // For inplace op we know direction of type alignment
    if (lhs.get_element_type().is_dynamic() || lhs.get_element_type() != rhs.get_element_type())
        rhs = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(rhs, lhs));
    OutputVector res = {context.mark_node(std::make_shared<T>(lhs, rhs))};
    return res;
}

inline OutputVector return_false_scalar(const NodeContext& context) {
    return {context.mark_node(ov::op::v0::Constant::create(element::boolean, Shape{}, {false}))};
}

inline OutputVector skip_node(const NodeContext& context) {
    return {context.get_input(0)};
}

}  // namespace op

}  // namespace jax
}  // namespace frontend
}  // namespace ov
