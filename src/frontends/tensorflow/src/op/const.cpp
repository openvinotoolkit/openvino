// Copyright (C) 2018-2023 Intel Corporation
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

OutputVector translate_const_op(const NodeContext& node) {
    auto tensor = node.get_attribute<ov::Tensor>("value");
    auto res = std::make_shared<ov::opset8::Constant>(tensor.get_element_type(), tensor.get_shape(), tensor.data());
    set_node_name(node.get_name(), res);
    return {res};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov