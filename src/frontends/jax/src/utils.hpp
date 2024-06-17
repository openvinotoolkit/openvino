// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/jax/node_context.hpp"

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

std::tuple<Output<Node>, Output<Node>> get_inputs_with_promoted_types(const NodeContext& context,
                                                                      size_t lhs_idx,
                                                                      size_t rhs_idx);

element::Type convert_dtype(int64_t pt_type);

namespace op {

template <typename T>
OutputVector translate_1to1_match_2_inputs(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    return {context.mark_node(std::make_shared<T>(context.get_input(0), context.get_input(1)))};
}

}  // namespace op

}  // namespace jax
}  // namespace frontend
}  // namespace ov
