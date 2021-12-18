// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/frontend/exception.hpp>

namespace ov {
namespace frontend {
namespace paddle {

class NodeContext;

class OpValidationFailurepaddle : public OpValidationFailure {
public:
    OpValidationFailurepaddle(const CheckLocInfo& check_loc_info,
                              const paddle::NodeContext& node,
                              const std::string& explanation)
        : OpValidationFailure(check_loc_info, get_error_msg_prefix_paddle(node), explanation) {}

private:
    static std::string get_error_msg_prefix_paddle(const paddle::NodeContext& node);
};
}  // namespace paddle
}  // namespace frontend

/// \brief Macro to check whether a boolean condition holds.
/// \param node_context Object of NodeContext class
/// \param cond Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evalutes to `false`.
/// \throws ::ov::OpValidationFailurepaddle if `cond` is false.
#define paddle_OP_CHECK(node_context, ...) \
    OPENVINO_ASSERT_HELPER(::ov::frontend::paddle::OpValidationFailurepaddle, (node_context), __VA_ARGS__)
}  // namespace ov
