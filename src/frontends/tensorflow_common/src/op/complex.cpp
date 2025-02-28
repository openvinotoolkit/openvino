// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "common_translators.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_complex_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Complex"}, true);
    auto tout = node.get_attribute<string>("Tout", "DT_COMPLEX64");
    element::Type complex_part_type = (tout == "DT_COMPLEX64" ? element::f32 : element::f64);

    auto complex_type_mark = common_translators::translate_complex(node);

    auto complex_type_mark_node = as_type_ptr<ComplexTypeMark>(complex_type_mark[0].get_node_shared_ptr());
    auto complex_tensor = complex_type_mark_node->get_data();
    if (complex_tensor.get_element_type() != complex_part_type) {
        complex_tensor = make_shared<v0::Convert>(complex_tensor, complex_part_type)->output(0);
        return make_shared<ComplexTypeMark>(complex_tensor, complex_part_type)->outputs();
    }
    return complex_type_mark;
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
