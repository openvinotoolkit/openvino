// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_translators.hpp"
#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/select.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace common_translators {

using namespace ov::op;
using namespace std;

OutputVector translate_angle(const NodeContext& context) {
    num_inputs_check(context, 1, 1, true);
    auto complex = context.get_input(0);

    auto op_type = context.get_op_type();
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(complex.get_node_shared_ptr());
    if (complex_type_mark) {
        // input is a complex tensor
        auto real_part = complex_type_mark->get_real();
        auto imag_part = complex_type_mark->get_imag();

        auto angle = translate_atan2_util(context, imag_part, real_part);
        return {angle};
    }

    // input real tensors is handled in such a way:
    // angle returns pi for negative real numbers,
    // zero for non-negative real numbers.
    const double pi_val = atan(1.0) * 4;
    auto real = complex;

    auto real_type = real.get_element_type();
    if (real_type.is_static() && real_type.is_integral()) {
        real = context.mark_node(std::make_shared<v0::Convert>(real, element::f32));
    }

    ov::Output<ov::Node> pi = create_same_type_const_scalar<double>(real, pi_val);
    ov::Output<ov::Node> zero = create_same_type_const_scalar<int32_t>(real, 0);
    auto is_negative = context.mark_node(make_shared<v1::Less>(real, zero));

    auto angle = context.mark_node(make_shared<v1::Select>(is_negative, pi, zero));
    return {angle};
}

}  // namespace common_translators
}  // namespace frontend
}  // namespace ov
