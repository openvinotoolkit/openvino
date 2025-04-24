// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/round.hpp"

#include "common_op_table.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_round_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Round", "ROUND"});

    auto input = node.get_input(0);
    // using default round mode "half_to_even" in openvino,
    // as TF has only that mode
    auto round_mode = v5::Round::RoundMode::HALF_TO_EVEN;
    auto res = make_shared<v5::Round>(input, round_mode);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
