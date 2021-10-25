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

OutputVector TranslateIdentityOp(const NodeContext& node) {
    auto input = node.get_ng_input(0);
    SetOutputName(node.get_name(), input);
    SetOutputName(node.get_name() + ":" + "0", input);
    return {input};
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov