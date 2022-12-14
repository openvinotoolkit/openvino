// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_ops/undefined_constant.hpp"
#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;
using namespace ov;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_const_op(const NodeContext& node) {
    auto ov_type = node.get_attribute<element::Type>("dtype");
    std::shared_ptr<op::Op> const_node;
    if (ov_type == element::undefined) {
        const_node = std::make_shared<UndefinedConstant>();
    } else {
        auto tensor = node.get_attribute<Tensor>("value");
        const_node = std::make_shared<opset8::Constant>(tensor.get_element_type(), tensor.get_shape(), tensor.data());
    }
    set_node_name(node.get_name(), const_node);
    return {const_node};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
