// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
OutputVector translate_rsub_common(const NodeContext& context,
                                   Output<Node> self,
                                   Output<Node> other,
                                   const Output<Node>& alpha) {
    if (alpha.get_node()) {
        // reverse aten::sub other - self * alpha
        auto alpha_casted = context.mark_node(std::make_shared<v1::ConvertLike>(alpha, self));
        self = context.mark_node(std::make_shared<v1::Multiply>(self, alpha_casted));
    }
    return {context.mark_node(std::make_shared<v1::Subtract>(other, self))};
}
}  // namespace

OutputVector translate_rsub(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    Output<Node> self;
    Output<Node> other;
    std::tie(self, other) = get_inputs_with_promoted_types(context, 0, 1);
    Output<Node> alpha;
    if (!context.input_is_none(2)) {
        alpha = context.get_input(2);
    }
    return translate_rsub_common(context, self, other, alpha);
};

OutputVector translate_rsub_fx(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    Output<Node> self;
    Output<Node> other;
    std::tie(self, other) = get_inputs_with_promoted_types(context, 0, 1);
    Output<Node> alpha;
    if (context.has_attribute("alpha")) {
        alpha = context.get_input("alpha");
    } else if (!context.input_is_none(2)) {
        alpha = context.get_input(2);
    }
    return translate_rsub_common(context, self, other, alpha);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
