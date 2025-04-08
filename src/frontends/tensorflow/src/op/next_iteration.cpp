// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_ops/next_iteration.hpp"

#include "common_op_table.hpp"
#include "helper_ops/merge.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_next_iteration_op(const NodeContext& node) {
    default_op_checks(node, 0, {"NextIteration"});

    auto next_iteration_node = make_shared<NextIteration>(node.get_decoder());
    set_node_name(node.get_name(), next_iteration_node);

    return next_iteration_node->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
