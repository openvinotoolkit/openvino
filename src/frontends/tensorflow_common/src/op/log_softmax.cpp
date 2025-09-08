// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/log_softmax.hpp"

#include "common_op_table.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_log_softmax_op(const NodeContext& node) {
    default_op_checks(node, 1, {"LogSoftmax", "LOG_SOFTMAX"});
    auto logits = node.get_input(0);
    auto log_softmax = make_shared<v5::LogSoftmax>(logits, -1);
    set_node_name(node.get_name(), log_softmax);
    return {log_softmax};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
