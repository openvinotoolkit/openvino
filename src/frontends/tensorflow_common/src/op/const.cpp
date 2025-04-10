// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/unsupported_constant.hpp"
#include "openvino/op/constant.hpp"

using namespace std;
using namespace ov::op;
using namespace ov;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_const_op(const NodeContext& node) {
    default_op_checks(node, 0, {"Const"});

    auto ov_type = node.get_attribute_as_any("dtype");
    std::shared_ptr<Node> const_node;
    if (!ov_type.is<ov::element::Type>() || ov_type.as<ov::element::Type>() == ov::element::dynamic) {
        const_node = std::make_shared<UnsupportedConstant>();
    } else {
        auto tensor = node.get_attribute<Tensor>("value");
        const_node = std::make_shared<v0::Constant>(tensor);
    }
    set_node_name(node.get_name(), const_node);
    return {const_node};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
