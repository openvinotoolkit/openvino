// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
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
    // Complex tensor is represented as a floating-point tensor of shape [N1, N2, ..., Nk, 2]
    // where real part is placed in the slice by last dimension [..., 0] and
    // imaginary part is placed by index [..., 1]
    int32_t axis_value = (op_type == "Real") ? 0 : 1;

    // check that complex type mark is set at the input
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
    TENSORFLOW_OP_VALIDATION(
        node,
        complex_type_mark,
        "[TensorFlow Frontend] internal error: ComplexTypeMark is not set at the input of " + op_type);
    auto data = complex_type_mark->input_value(0);

    // gather the required slice corresponding to Real or Imaginary part
    auto gather_index = make_shared<v0::Constant>(element::i32, Shape{}, axis_value);
    auto gather_axis = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
    auto complex_part = make_shared<v8::Gather>(data, gather_index, gather_axis)->output(0);

    // align output type required by tout attribute
    complex_part = make_shared<v0::Convert>(complex_part, tout);

    set_node_name(node.get_name(), complex_part.get_node_shared_ptr());

    return {complex_part};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
