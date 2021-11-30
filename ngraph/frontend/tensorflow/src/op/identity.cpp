// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tf {
namespace op {

OutputVector translate_identity_op(const NodeContext& node) {
    auto input = node.get_input(0);
    set_out_name(node.get_name(), input);
    set_out_name(node.get_name() + ":" + "0", input);
    return {input};
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov