// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_reverse_sequence_op(const NodeContext& node) {
    default_op_checks(node, 2, {"ReverseSequence"});
    auto input = node.get_input(0);
    auto seq_lengths = node.get_input(1);

    // retrieve attributes
    auto seq_axis = node.get_attribute<int64_t>("seq_dim");
    auto batch_axis = node.get_attribute<int64_t>("batch_dim", 0);

    auto reverse_sequence = make_shared<ReverseSequence>(input, seq_lengths, batch_axis, seq_axis);
    set_node_name(node.get_name(), reverse_sequence);
    return {reverse_sequence};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
