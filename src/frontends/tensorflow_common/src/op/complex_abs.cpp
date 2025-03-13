// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_complex_abs_op(const NodeContext& node) {
    default_op_checks(node, 1, {"ComplexAbs"}, true);
    auto op_type = node.get_op_type();
    auto x = node.get_input(0);
    auto tout = node.get_attribute<element::Type>("Tout", element::f32);

    auto complex_abs = ComplexTypeMark::abs(node, x);

    // align output type required by tout attribute
    complex_abs = make_shared<v0::Convert>(complex_abs, tout);

    set_node_name(node.get_name(), complex_abs.get_node_shared_ptr());
    return {complex_abs};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
