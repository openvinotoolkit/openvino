// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/frontend_exceptions.hpp>

namespace ov {
namespace frontend {
namespace paddlepaddle {

class NodeContext;

class OpValidationFailurePDPD : public OpValidationFailure {
public:
    OpValidationFailurePDPD(const CheckLocInfo& check_loc_info,
                            const paddlepaddle::NodeContext& node,
                            const std::string& explanation)
        : OpValidationFailure(check_loc_info, get_error_msg_prefix_pdpd(node), explanation) {}

private:
    static std::string get_error_msg_prefix_pdpd(const paddlepaddle::NodeContext& node);
};
}  // namespace paddlepaddle
}  // namespace frontend

/// \brief Macro to check whether a boolean condition holds.
/// \param node_context Object of NodeContext class
/// \param cond Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evalutes to `false`.
/// \throws ::ov::OpValidationFailurePDPD if `cond` is false.
#define PADDLEPADDLE_OP_CHECK(node_context, ...) \
    OPENVINO_ASSERT_HELPER(::ov::frontend::paddlepaddle::OpValidationFailurePDPD, (node_context), __VA_ARGS__)
}  // namespace ov
