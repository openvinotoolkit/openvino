// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/string_constant.hpp"
#include "helper_ops/unsupported_constant.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;
using namespace ov;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_const_op(const NodeContext& node) {
    auto ov_type = node.get_attribute_as_any("dtype");
    std::shared_ptr<Node> const_node;
    if (!ov_type.is<ov::element::Type>() || ov_type.as<ov::element::Type>() == ov::element::dynamic ||
        ov_type.as<ov::element::Type>() == ov::element::undefined) {
        if (ov_type.is<std::string>() && ov_type.as<std::string>() == "DT_STRING") {
            const_node = std::make_shared<StringConstant>(node.get_attribute_as_any("value"));
        } else {
            const_node = std::make_shared<UnsupportedConstant>();
        }
    } else {
        auto tensor = node.get_attribute<Tensor>("value");
        const_node = std::make_shared<Constant>(tensor);
    }
    set_node_name(node.get_name(), const_node);
    return {const_node};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
