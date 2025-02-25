// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_translators.hpp"
#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/select.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace common_translators {

using namespace ov::op;
using namespace ov::frontend;
using namespace std;
using namespace ov;

OutputVector translate_angle(const NodeContext& context) {
    num_inputs_check(context, 1, 1, true);
    auto complex = context.get_input(0);

    auto op_type = context.get_op_type();
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(complex.get_node_shared_ptr());
    if (complex_type_mark) {
        // input is a complex tensor
        complex = complex_type_mark->input_value(0);

        auto real_part = ComplexTypeMark::get_real_part(context, complex);
        auto imag_part = ComplexTypeMark::get_imag_part(context, complex);

        auto angle = translate_atan2_util(context, imag_part, real_part);
        return {angle};
    }

    // input real tensors is handled in such a way:
    // angle returns pi for negative real numbers,
    // zero for non-negative real numbers.
    const double pi_val = atan(1.0) * 4;
    auto real = complex;

    auto zero_as_input = context.mark_node(make_shared<v0::Constant>(element::i32, Shape{}, 0));
    zero_as_input = context.mark_node(make_shared<v1::ConvertLike>(zero_as_input, real));
    auto is_negative = context.mark_node(make_shared<v1::Less>(real, zero_as_input));

    ov::Output<ov::Node> pi;
    ov::Output<ov::Node> zero;
    if (real.get_element_type() == ov::element::f64) {
        pi = context.mark_node(make_shared<v0::Constant>(ov::element::f64, Shape{}, pi_val));
        zero = context.mark_node(make_shared<v0::Constant>(ov::element::f64, Shape{}, 0));
    } else {
        pi = context.mark_node(make_shared<v0::Constant>(ov::element::f32, Shape{}, static_cast<float>(pi_val)));
        zero = context.mark_node(make_shared<v0::Constant>(ov::element::f32, Shape{}, 0));
    }

    auto angle = context.mark_node(make_shared<v1::Select>(is_negative, pi, zero));
    return {angle};
}

}  // namespace common_translators
}  // namespace frontend
}  // namespace ov
