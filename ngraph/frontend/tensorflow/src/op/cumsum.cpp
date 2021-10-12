// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <openvino/opsets/opset8.hpp>

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateCumsumOp(const NodeContext& node) {
    auto ng_x = node.get_ng_input(0), ng_axis = node.get_ng_input(1);
    auto exclusive = node.get_attribute<bool>("exclusive"), reverse = node.get_attribute<bool>("reverse");

    return {ConstructNgNode<CumSum>(node.get_name(), ng_x, ng_axis, exclusive, reverse)};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
