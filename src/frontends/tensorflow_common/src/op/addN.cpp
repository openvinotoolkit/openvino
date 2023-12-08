// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/add.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_add_n_op(const NodeContext& node) {
    default_op_checks(node, 1, {"AddN", "ADD_N"});
    int num_size = static_cast<int>(node.get_input_size());

    Output<Node> result = node.get_input(0);
    for (int ind = 1; ind < num_size; ++ind) {
        result = make_shared<v1::Add>(result, node.get_input(ind));
    }

    set_node_name(node.get_name(), result.get_node_shared_ptr());
    return {result};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
