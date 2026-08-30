// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/prelu.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_rrelu(const NodeContext& context) {
    // aten::rrelu(Tensor self, Scalar lower=0.125, Scalar upper=0.3333, bool training=False,
    //             Generator? generator=None) -> Tensor
    // In inference mode the negative slope is a fixed value: (lower + upper) / 2.
    num_inputs_check(context, 1, 5);
    auto x = context.get_input(0);

    float lower = 1.0f / 8;
    float upper = 1.0f / 3;

    if (context.get_input_size() > 1 && !context.input_is_none(1)) {
        lower = context.const_input<float>(1);
    }
    if (context.get_input_size() > 2 && !context.input_is_none(2)) {
        upper = context.const_input<float>(2);
    }

    const float slope = (lower + upper) / 2.0f;
    Output<Node> negative_slope = ov::op::v0::Constant::create(element::f32, Shape{1}, {slope});
    negative_slope = context.mark_node(std::make_shared<v1::ConvertLike>(negative_slope, x));

    return {context.mark_node(std::make_shared<v0::PRelu>(x, negative_slope))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
