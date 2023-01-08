// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_round_op(const NodeContext& node) {
    auto input = node.get_input(0);
    // using default round mode "half_to_even" in openvino,
    // as TF has only that mode
    auto round_mode = Round::RoundMode::HALF_TO_EVEN;
    auto res = make_shared<Round>(input, round_mode);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
