// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "utils.hpp"

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
