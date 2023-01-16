// Copyright (C) 2018-2022 Intel Corporation
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

ov::OutputVector translate_rank_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Rank"});
    auto input = node.get_input(0);
    auto input_shape = make_shared<ShapeOf>(input, ov::element::i32);
    auto unsqueeze_input_rank = make_shared<ShapeOf>(input_shape, ov::element::i32);
    auto input_rank = make_shared<Squeeze>(unsqueeze_input_rank);
    set_node_name(node.get_name(), input_rank);
    return {input_rank};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
