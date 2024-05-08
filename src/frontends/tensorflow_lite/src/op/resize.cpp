// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/frontend/tensorflow_lite/node_context.hpp"

using namespace std;
using namespace ov::frontend::tensorflow::op;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector resize_bilinear(const ov::frontend::tensorflow_lite::NodeContext& node) {
    return translate_interpolate_op(node);
}

OutputVector resize_nearest_neightbor(const ov::frontend::tensorflow_lite::NodeContext& node) {
    return translate_interpolate_op(node);
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
