// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/multinomial.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_multinomial(const NodeContext& context) {
    num_inputs_check(context, 3, 5);
    auto input = context.get_input(0);
    auto num_samples = context.get_input(1);
    auto replacement = context.const_input<bool>(2);
    FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(3),
                                  "aten::multinomial conversion with generator does not supported");
    auto multinomial =
        context.mark_node(std::make_shared<v13::Multinomial>(input, num_samples, element::i64, replacement, false));
    if (!context.input_is_none(5)) {
        context.mutate_input(5, multinomial);
    }
    return {multinomial};
};
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
