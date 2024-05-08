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

OutputVector reverse_sequence(const ov::frontend::tensorflow_lite::NodeContext& node) {
    return translate_reverse_sequence_op(node);
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
