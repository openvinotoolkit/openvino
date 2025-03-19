// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_tensor_scatter_update_op(const NodeContext& node) {
    default_op_checks(node, 3, {"TensorScatterUpdate"});
    auto tensor = node.get_input(0);
    auto indices = node.get_input(1);
    auto updates = node.get_input(2);

    auto scatter_nd = make_shared<v15::ScatterNDUpdate>(tensor, indices, updates);
    set_node_name(node.get_name(), scatter_nd);
    return {scatter_nd};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
