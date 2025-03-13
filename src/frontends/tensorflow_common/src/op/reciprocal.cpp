// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/power.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_reciprocal_op(const NodeContext& node) {
    // computes element-wise 1/x, where x - input
    default_op_checks(node, 1, {"Reciprocal"}, true);
    auto x = node.get_input(0);
    auto complex_type_mark_x = as_type_ptr<ComplexTypeMark>(x.get_node_shared_ptr());
    if (complex_type_mark_x) {
        x = complex_type_mark_x->get_data();
        auto minus_one = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
        auto two = create_same_type_const_scalar<int32_t>(x, 2);
        auto gather_index_real = make_shared<v0::Constant>(element::i32, Shape{1}, 0);
        auto gather_index_imag = make_shared<v0::Constant>(element::i32, Shape{1}, 1);
        auto x_real = make_shared<v8::Gather>(x, gather_index_real, minus_one)->output(0);
        auto x_imag = make_shared<v8::Gather>(x, gather_index_imag, minus_one)->output(0);

        // compute (a^2+b^2)
        auto real_squared_norm = make_shared<v1::Power>(x_real, two);
        auto img_squared_norm = make_shared<v1::Power>(x_imag, two);
        auto squared_norm = make_shared<v1::Add>(real_squared_norm, img_squared_norm);

        // compute 1/(a+bi) = (a-bi)/(a^2+b^2)
        auto complex_reciprocal = make_shared<v1::Divide>(
            make_shared<v0::Concat>(OutputVector{x_real, make_shared<ov::op::v0::Negative>(x_imag)}, -1),
            squared_norm);
        auto complex_result =
            make_shared<ComplexTypeMark>(complex_reciprocal, complex_type_mark_x->get_complex_part_type());
        set_node_name(node.get_name(), complex_reciprocal);
        return {complex_result};
    }

    // For real numbers, computes element-wise 1/x, where x - input
    auto minus_one_const = create_same_type_const_scalar<int32_t>(x, -1);
    auto reciprocal = make_shared<v1::Power>(x, minus_one_const);
    set_node_name(node.get_name(), reciprocal);
    return {reciprocal};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
