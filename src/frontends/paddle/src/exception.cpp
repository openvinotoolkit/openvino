// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/exception.hpp"

#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
std::string OpValidationFailure::get_error_msg_prefix_paddle(const paddle::NodeContext& node) {
    std::stringstream ss;
    ss << "While validating node '" << node.get_op_type() << '\'';
    return ss.str();
}

void throw_op_validation_failure(const CheckLocInfo& check_loc_info,
                                 const ov::frontend::paddle::NodeContext& context_info,
                                 const std::string& explanation) {
    throw ov::frontend::paddle::OpValidationFailure(check_loc_info, context_info, explanation);
}
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
