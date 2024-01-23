// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/frontend/tensorflow/variable.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_readvariable_op(const NodeContext& node) {
    default_op_checks(node, 1, {"ReadVariableOp"});

    // get_input will care of reading variable value
    auto variable_value = node.get_input(0);
    set_node_name(node.get_name(), variable_value.get_node_shared_ptr());
    return {variable_value};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
