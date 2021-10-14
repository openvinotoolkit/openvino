// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <openvino/opsets/opset8.hpp>

#include "node_context.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateEluOp(const NodeContext& node) {
    auto input = node.get_ng_input(0);
    auto alpha = 1.0;  // node.get_attribute<float>("alpha");
    return {ConstructNgNode<Elu>(node.get_name(), input, alpha)};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
