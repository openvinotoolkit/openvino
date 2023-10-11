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

void OpValidationFailure::create(const CheckLocInfo& check_loc_info,
                                 const NodeContext& node,
                                 const std::string& explanation) {
    throw OpValidationFailure(make_what(check_loc_info, get_error_msg_prefix_paddle(node), explanation));
}
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
