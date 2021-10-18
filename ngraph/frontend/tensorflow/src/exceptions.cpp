// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "exceptions.hpp"
#include "node_context.hpp"

namespace ov {
namespace frontend {
namespace tf {
std::string OpValidationFailureTF::get_error_msg_prefix_tf(const tf::NodeContext& node) {
    std::stringstream ss;
    ss << "While validating node '" << node.get_op_type() << '\'';
    return ss.str();
}
}  // namespace tf
}  // namespace frontend
}  // namespace ov
