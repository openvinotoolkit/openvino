// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "input_model.hpp"
#include "openvino/op/parameter.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;
using namespace ov;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_placeholder_linked_op(const NodeContext& node) {
    default_op_checks(node, 0, {"Placeholder"});
    auto dtype = node.get_attribute<element::Type>("dtype");
    auto shape = node.get_attribute<PartialShape>("shape", PartialShape::dynamic());
    auto res = std::make_shared<v0::Parameter>(dtype, shape);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
