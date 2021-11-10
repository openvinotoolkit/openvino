// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Please review the entire file. These changes have been merged without a detailed review.
// This comment will be removed after that.

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tf {
namespace op {

OutputVector translate_placeholder_op(const NodeContext& node) {
    auto ng_et = node.get_attribute<ov::element::Type>("dtype");
    auto ng_shape = node.get_attribute<ov::PartialShape>("shape", ov::PartialShape());

    auto res = std::make_shared<Parameter>(ng_et, ng_shape);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov