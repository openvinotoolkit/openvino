// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/is_nan.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

namespace {
OutputVector translate_xlogy_common(const NodeContext& context, bool inplace) {
    // aten::xlogy(Tensor self, Tensor other, *, Tensor? out=None) -> Tensor
    // aten::xlogy_(Tensor(a!) self, Tensor other) -> Tensor(a!)
    // xlogy(x, y) = 0 when x == 0 and y is not NaN; NaN when y is NaN; otherwise x * log(y)
    num_inputs_check(context, 2, 3);
    Output<Node> x;
    Output<Node> y;
    std::tie(x, y) = get_inputs_with_promoted_types(context, 0, 1);

    // Promotion narrows a python scalar to the integer tensor type (a uint8 tensor with 256 gives 0), so the
    // floating computation casts the original operands and uses the promoted inputs only to pick the type.
    const auto x_in = context.get_input(0);
    const auto y_in = context.get_input(1);

    // torch.xlogy returns the default floating point type for integer and boolean inputs
    const auto dtype = x.get_element_type();
    if (dtype.is_static() && !dtype.is_real()) {
        auto float_type = element::f32;
        const auto out_type = simplified_type_interpret(context.get_output_type(0));
        if (out_type.is<element::Type>() && out_type.as<element::Type>().is_real()) {
            float_type = out_type.as<element::Type>();
        }
        x = context.mark_node(std::make_shared<v0::Convert>(x_in, float_type));
        y = context.mark_node(std::make_shared<v0::Convert>(y_in, float_type));
    } else if (dtype.is_dynamic()) {
        // The input type is not known yet (e.g. scripted model before input types are set). Promote against a
        // floating python scalar in the graph so integer and boolean inputs resolve to f32 once types are known,
        // while floating inputs keep their own type.
        auto float_scalar = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0.5}))->output(0);
        align_eltwise_input_types(context, x, float_scalar, false, true);
        y = context.mark_node(std::make_shared<v1::ConvertLike>(y_in, x));
        x = context.mark_node(std::make_shared<v1::ConvertLike>(x_in, x));
    }

    auto zero = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    zero = context.mark_node(std::make_shared<v1::ConvertLike>(zero, x));

    const auto log_y = context.mark_node(std::make_shared<v0::Log>(y));
    const auto x_log_y = context.mark_node(std::make_shared<v1::Multiply>(x, log_y));
    const auto x_is_zero = context.mark_node(std::make_shared<v1::Equal>(x, zero));
    const auto y_is_nan = context.mark_node(std::make_shared<v10::IsNaN>(y));
    const auto y_not_nan = context.mark_node(std::make_shared<v1::LogicalNot>(y_is_nan));
    const auto use_zero = context.mark_node(std::make_shared<v1::LogicalAnd>(x_is_zero, y_not_nan));
    Output<Node> result = context.mark_node(std::make_shared<v1::Select>(use_zero, zero, x_log_y));

    if (inplace) {
        // In-place xlogy keeps the dtype of self: float32 self with float64 other computes in float64 and casts back.
        result = context.mark_node(std::make_shared<v1::ConvertLike>(result, x_in));
        context.mutate_input(0, result);
    } else if (!context.input_is_none(2)) {
        // out= keeps the dtype of the destination: float64 operands written to a float32 out are rounded to float32.
        result = context.mark_node(std::make_shared<v1::ConvertLike>(result, context.get_input(2)));
        context.mutate_input(2, result);
    }
    return {result};
}
}  // namespace

OutputVector translate_xlogy(const NodeContext& context) {
    return translate_xlogy_common(context, false);
}

OutputVector translate_xlogy_(const NodeContext& context) {
    return translate_xlogy_common(context, true);
}

}  // namespace ov::frontend::pytorch::op
