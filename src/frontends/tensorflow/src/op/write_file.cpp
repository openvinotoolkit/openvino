// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/keep_in_graph_op.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "tf_utils.hpp"
#include "utils.hpp"

using namespace std;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_write_file(const NodeContext& node) {
    default_op_checks(node, 1, {"WriteFile"});
    OutputVector ov_inputs = {node.get_input(0)};

    auto write_file_node = make_shared<KeepInGraphOp>("WriteFile", ov_inputs);
    set_node_name(node.get_name(), write_file_node);

    // Original WriteFile op has no outputs, but we need to return one auxiliary output
    // to keep this operation in graph, otherwise we will have no access to operation from translator.
    // KeepInGraphOp is added to Sink vector in ov::Model and is kept in the graph until the end of conversion.
    return {write_file_node};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov