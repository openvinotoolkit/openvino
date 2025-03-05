// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_inv_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Inv"}, true);
    auto x = node.get_input(0);
    auto inv = ComplexTypeMark::inv(node, x);
    set_node_name(node.get_name(), inv.get_node_shared_ptr());
    return {inv};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
