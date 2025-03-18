// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/exp.hpp"

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_exp(const NodeContext& context) {
    num_inputs_check(context, 1, 2, true);

    auto x = context.get_input(0);
    // This const only needed for type alignment
    auto dummy_const = context.mark_node(v0::Constant::create(element::f32, Shape({}), {0.5}))->output(0);
    align_eltwise_input_types(context, x, dummy_const, false, true);

    auto exp = ComplexTypeMark::exp(context, x);

    return {exp};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov