// Copyright (C) 2018-2023 Intel Corporation
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

    // check that complex type mark is set to the input
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(x.get_node_shared_ptr());
    TENSORFLOW_OP_VALIDATION(node,
                             complex_type_mark,
                             "[TensorFlow Frontend] internal error: ComplexTypeMark is not set to input of " + op_type);
    auto complex_part_type = complex_type_mark->get_complex_part_type();
    // data is complex tensor representation in a form [N1, N2, ..., Nk, 2]
    // where slice [N1, N2, ..., Nk, 0] contains real part of the complex tensor and
    // slice [N1, N2, ..., Nk, 1] contains imaginary part of the complex tensor
    auto data = complex_type_mark->input_value(0);

    // compute element-wise square for complex representation
    auto const_two = make_shared<v0::Constant>(complex_part_type, Shape{}, 2);
    auto squared_data = make_shared<v1::Power>(data, const_two);

    // compute sum of squared real and imaginary parts
    auto const_minus_one = make_shared<v0::Constant>(element::i32, Shape{}, -1);
    auto complex_abs = make_shared<v1::ReduceSum>(squared_data, const_minus_one, false)->output(0);

    // compute ComplexAbs by root-squared operation
    auto const_half = make_shared<v0::Constant>(complex_part_type, Shape{}, 0.5f);
    complex_abs = make_shared<v1::Power>(complex_abs, const_half);

    // aling output type required by tout attribute
    complex_abs = make_shared<v0::Convert>(complex_abs, tout);

    set_node_name(node.get_name(), complex_abs.get_node_shared_ptr());
    return {complex_abs};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
