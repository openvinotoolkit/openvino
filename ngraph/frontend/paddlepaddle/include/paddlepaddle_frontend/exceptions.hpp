// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend_exceptions.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {

class NodeContext;

class CheckFailurePDPD : public CheckFailureFrontEnd {
public:
    CheckFailurePDPD(const ErrorCode error_code, const CheckLocInfo &check_loc_info, const std::string &context, const std::string &explanation)
            : CheckFailureFrontEnd(error_code, check_loc_info, " \nPaddlePaddle FrontEnd failed" + context, explanation) {
    }
};

class NodeValidationFailurePDPD : public CheckFailurePDPD {
public:
    NodeValidationFailurePDPD(const ErrorCode error_code,
                              const CheckLocInfo &check_loc_info,
                              const pdpd::NodeContext &node,
                              const std::string &explanation)
            : CheckFailurePDPD(error_code, check_loc_info, get_error_msg_prefix_pdpd(node), explanation) {
    }

private:
    static std::string get_error_msg_prefix_pdpd(const pdpd::NodeContext &node);
};
} // namespace pdpd
} // namespace frontend

/// \brief Macro to check whether a boolean condition holds.
/// \param error_code Additional indicator of the type of error.
/// \param node_context Object of NodeContext class
/// \param cond Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evalutes to `false`.
/// \throws ::ngraph::CheckFailurePDPD if `cond` is false.
#define PDPD_NODE_VALIDATION_CHECK(error_code, node_context, ...) \
        FRONT_END_CHECK_HELPER(error_code, ::ngraph::frontend::pdpd::NodeValidationFailurePDPD, (node_context), __VA_ARGS__)

/// \brief Macro to check whether a boolean condition holds.
/// \param error_code Additional indicator of the type of error.
/// \param cond Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evalutes to `false`.
/// \throws ::ngraph::CheckFailurePDPD if `cond` is false.
#define PDPD_CHECK(error_code, ...) FRONT_END_CHECK_HELPER(error_code, ::ngraph::frontend::pdpd::CheckFailurePDPD, "", __VA_ARGS__)

#define PDPD_NOT_IMPLEMENTED(msg) PDPD_CHECK(::ngraph::frontend::ErrorCode::NOT_IMPLEMENTED, \
        false, std::string(msg) + " is not implemented")

} // namespace ngraph

