// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

ov::OutputVector translate_shape_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Shape", "SHAPE"});
    auto input = node.get_input(0);
    auto out_type = node.get_attribute<element::Type>("out_type", element::i32);
    auto shapeof = make_shared<ShapeOf>(input, out_type);
    set_node_name(node.get_name(), shapeof);
    return {shapeof};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
