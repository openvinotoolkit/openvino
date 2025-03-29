// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/constant.hpp"
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

void num_inputs_check(const NodeContext& context, size_t min_inputs);

void add_exception_to_fw_node(std::shared_ptr<Node> node, const std::string& msg);

bool is_python_scalar_input(const NodeContext& context, size_t index);

element::Type convert_dtype(int64_t pt_type);

OutputVector make_framework_node(const NodeContext& context, const std::string& exception);

namespace op {
template <typename T>
OutputVector translate_1to1_match_1_input(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    return {std::make_shared<T>(context.get_input(0))};
}

template <typename T>
OutputVector translate_1to1_match_2_inputs(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    return {std::make_shared<T>(context.get_input(0), context.get_input(1))};
}

inline OutputVector skip_node(const NodeContext& context) {
    return {context.get_input(0)};
}

template <typename T>
ov::Output<ov::Node> create_same_type_const_scalar(const ov::Output<ov::Node>& same_type_output, const T& value) {
    if (same_type_output.get_element_type().is_static()) {
        return std::make_shared<ov::op::v0::Constant>(same_type_output.get_element_type(), ov::Shape{}, value);
    } else {
        ov::Output<ov::Node> const_res =
            std::make_shared<ov::op::v0::Constant>(ov::element::from<T>(), ov::Shape{}, value);
        const_res = std::make_shared<ov::op::v1::ConvertLike>(const_res, same_type_output);
        return const_res;
    }
}

}  // namespace op

}  // namespace jax
}  // namespace frontend
}  // namespace ov
