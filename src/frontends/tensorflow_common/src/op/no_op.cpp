// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_no_op(const NodeContext& node) {
    // the operation does nothing in terms of data generation
    default_op_checks(node, 0, {"NoOp", "SaveV2"});
    return {};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
