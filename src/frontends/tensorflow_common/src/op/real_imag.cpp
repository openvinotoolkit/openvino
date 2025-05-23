// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "common_translators.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_real_imag_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Real", "Imag"}, true);
    auto op_type = node.get_op_type();
    auto input = node.get_input(0);
    auto tout = node.get_attribute<element::Type>("Tout", element::f32);

    ov::Output<ov::Node> complex_part;
    if (node.get_op_type() == "Real") {
        complex_part = common_translators::translate_real(node)[0];
    } else {
        complex_part = common_translators::translate_imag(node)[0];
    }

    // align output type required by tout attribute
    if (tout != complex_part.get_element_type()) {
        complex_part = make_shared<v0::Convert>(complex_part, tout);
    }

    set_node_name(node.get_name(), complex_part.get_node_shared_ptr());

    return {complex_part};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
