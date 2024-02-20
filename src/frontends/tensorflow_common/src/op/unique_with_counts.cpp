// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/unique.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_unique_with_counts_op(const NodeContext& node) {
    default_op_checks(node, 1, {"UniqueWithCounts"});

    // get input 'x' from node and node name
    auto x = node.get_input(0);
    auto node_name = node.get_name();

    auto unique = make_shared<v10::Unique>(x, false, ov::element::i32, ov::element::i32);
    set_node_name(node_name, unique);

    // story 'y', 'idx', and 'count' outputs from Unique in separate variables
    auto y = unique->output(0);
    auto idx = unique->output(2);
    auto count = unique->output(3);

    return {{y}, {idx}, {count}};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov