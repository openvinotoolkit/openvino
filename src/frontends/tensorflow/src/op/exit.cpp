// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_ops/exit.hpp"

#include "common_op_table.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_exit_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Exit"});
    auto data = node.get_input(0);

    auto exit_node = make_shared<Exit>(data, node.get_decoder());
    set_node_name(node.get_name(), exit_node);

    return exit_node->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
