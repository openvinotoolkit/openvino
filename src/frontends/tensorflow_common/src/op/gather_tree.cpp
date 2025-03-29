// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather_tree.hpp"

#include "common_op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_gather_tree_op(const NodeContext& node) {
    default_op_checks(node, 4, {"GatherTree", "Addons>GatherTree"});
    auto step_ids = node.get_input(0);
    auto parent_ids = node.get_input(1);
    auto max_sequence_lengths = node.get_input(2);
    auto end_token = node.get_input(3);

    // adjust end_token that must be a scalar
    auto new_shape_end_token = make_shared<v0::Constant>(element::i32, Shape{0}, vector<int32_t>{});
    end_token = make_shared<v1::Reshape>(end_token, new_shape_end_token, false);

    auto gather_tree = make_shared<v1::GatherTree>(step_ids, parent_ids, max_sequence_lengths, end_token);
    set_node_name(node.get_name(), gather_tree);

    return {gather_tree};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
