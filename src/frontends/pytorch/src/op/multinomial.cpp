// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/multinomial.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_multinomial(const NodeContext& context) {
    num_inputs_check(context, 3, 5);
    auto const_neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto input = context.get_input(0);
    auto num_samples = context.mark_node(std::make_shared<v1::Reshape>(get_input_as_i32(context, 1), const_1, false));
    auto replacement = context.const_input<bool>(2);
    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(3),
                                "aten::multinomial conversion with generator is not supported");

    // Torch multinomial accept input of [class_probs] or [bs, class_probs], convert always to [bs, class_probs] for OV.
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    auto class_probs_shape = context.mark_node(std::make_shared<v8::Gather>(input_shape, const_neg_1, const_0));
    auto inp_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{const_neg_1, class_probs_shape}, 0));
    input = context.mark_node(std::make_shared<v1::Reshape>(input, inp_shape, false));

    auto multinomial =
        context.mark_node(std::make_shared<v13::Multinomial>(input, num_samples, element::i64, replacement, false));

    // Torch multinomial can return [num_samples] or [bs, num_samples] based on input dim, reshape to correct shape.
    auto out_shape = context.mark_node(
        std::make_shared<v12::ScatterElementsUpdate>(input_shape, const_neg_1, num_samples, const_neg_1));
    multinomial = context.mark_node(std::make_shared<v1::Reshape>(multinomial, out_shape, false));

    if (!context.input_is_none(5)) {
        context.mutate_input(5, multinomial);
    }
    return {multinomial};
};
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
