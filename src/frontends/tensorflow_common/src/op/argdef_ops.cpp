// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_input_arg_op(const NodeContext& node) {
    default_op_checks(node, 0, {"input_arg"});
    auto param_type = node.get_attribute<element::Type>("type");

    auto param = std::make_shared<v0::Parameter>(param_type, PartialShape::dynamic());
    set_node_name(node.get_name(), param);
    return param->outputs();
}

OutputVector translate_output_arg_op(const NodeContext& node) {
    default_op_checks(node, 1, {"output_arg"});
    auto result = std::make_shared<v0::Result>(node.get_input(0));
    set_node_name(node.get_name(), result);
    return result->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
