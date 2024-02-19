// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_scatter_nd_op(const NodeContext& node) {
    default_op_checks(node, 3, {"ScatterNd", "SCATTER_ND"});
    auto input_indices = node.get_input(0);
    auto updates = node.get_input(1);
    auto shape = node.get_input(2);

    auto input_data = create_same_type_const<int32_t>(updates, vector<int32_t>{0}, Shape{1});
    auto broadcast = make_shared<v3::Broadcast>(input_data, shape);
    auto scatter_nd = make_shared<v3::ScatterNDUpdate>(broadcast, input_indices, updates);
    set_node_name(node.get_name(), scatter_nd);
    return {scatter_nd};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
