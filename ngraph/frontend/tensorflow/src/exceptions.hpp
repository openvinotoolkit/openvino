// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <frontend_manager/frontend_exceptions.hpp>
#include <openvino/core/node.hpp>

namespace ov {
namespace frontend {
namespace tf {

class NodeContext;

class OpValidationFailureTF : public ngraph::frontend::OpValidationFailure {
public:
    OpValidationFailureTF(const CheckLocInfo& check_loc_info, const NodeContext& node, const std::string& explanation)
        : OpValidationFailure(check_loc_info, get_error_msg_prefix_tf(node), explanation) {}

private:
    static std::string get_error_msg_prefix_tf(const NodeContext& node);
};
}  // namespace tf
}  // namespace frontend

/// \brief Macro to check whether a boolean condition holds.
/// \param node_context Object of NodeContext class
/// \param cond Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evalutes to `false`.
/// \throws ::ov::OpValidationFailureTF if `cond` is false.
#define TF_OP_VALIDATION_CHECK(node_context, ...) \
    NGRAPH_CHECK_HELPER(::ov::frontend::tf::OpValidationFailureTF, (node_context), __VA_ARGS__)
}  // namespace ov
