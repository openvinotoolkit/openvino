// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_conv_2d_op(const NodeContext& node) {
    return translate_convolution_op(node, 2);
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
