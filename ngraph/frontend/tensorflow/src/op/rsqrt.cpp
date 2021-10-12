// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateRsqrtOp(const NodeContext& node) {
    auto input = node.get_ng_input(0);
    auto ng_exponent = ConstructNgNode<Constant>(node.get_name(), input.get_element_type(), Shape{1}, -0.5f);
    auto power = make_shared<Power>(input, ng_exponent);
    power->set_friendly_name(node.get_name());
    return power->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph
