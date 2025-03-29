// Copyright (C) 2018-2025 Intel Corporation
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

void OpValidationFailure::create(const char* file,
                                 int line,
                                 const char* check_string,
                                 const NodeContext& node,
                                 const std::string& explanation) {
    throw OpValidationFailure(make_what(file, line, check_string, get_error_msg_prefix_paddle(node), explanation));
}
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
