// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset8;
using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
ov::OutputVector translate_roll_op(const NodeContext& node) {
    auto data = node.get_input(0);
    auto shift = node.get_input(1);
    auto axis = node.get_input(2);
    auto res = std::make_shared<Roll>(data, shift, axis);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
