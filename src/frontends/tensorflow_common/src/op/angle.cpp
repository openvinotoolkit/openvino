// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "common_op_table.hpp"
#include "common_translators.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_angle_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Angle"}, true);
    auto result_type = node.get_attribute<ov::element::Type>("Tout", element::f32);

    auto angle = common_translators::translate_angle(node)[0];

    angle = make_shared<v0::Convert>(angle, result_type)->output(0);

    set_node_name(node.get_name(), angle.get_node_shared_ptr());
    return {angle};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
