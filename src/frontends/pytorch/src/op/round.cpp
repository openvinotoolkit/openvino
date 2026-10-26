// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/round.hpp"

#include <cmath>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/convert_promote_types.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "transformations/rt_info/nonconvertible_divide.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

namespace {
// torch.round(x, decimals=d) is nearbyint(x * 10^d) / 10^d for d >= 0 and
// nearbyint(x / 10^-d) * 10^-d for d < 0, where nearbyint rounds half to even. The negative case
// is not folded into the positive one: torch divides before rounding it there, and the two forms
// are not bit-identical in floating point.
Output<Node> round_with_decimals(const NodeContext& context, const Output<Node>& data, int64_t decimals) {
    // The kernel is evaluated in opmath_t, so half and bfloat16 inputs are widened to float before
    // the scaling: scaling in the low precision type itself overflows instead of rounding
    // (f16 1000 with decimals = 2 becomes inf, while torch keeps 1000). The direct frontend path
    // specializes input types only after conversion, so use ConvertPromoteTypes with an f32 anchor
    // for dynamic inputs. Its output stays dynamic during translation, then resolves to f32 for
    // f16/bf16, remains f32 for f32, and remains f64 for f64. ConvertLike below restores the input
    // type after the calculation.
    const auto data_type = data.get_element_type();
    const bool widen = data_type.is_dynamic() || data_type == element::f16 || data_type == element::bf16;
    Output<Node> x = data;
    if (data_type.is_dynamic()) {
        auto anchor = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
        x = context.mark_node(std::make_shared<v14::ConvertPromoteTypes>(data, anchor))->output(0);
    } else if (data_type == element::f16 || data_type == element::bf16) {
        x = context.mark_node(std::make_shared<v0::Convert>(data, element::f32));
    }

    // The factor is computed in fp64 and materialized in the working type: the input element type
    // may still be dynamic during conversion, so it cannot be used to type a constant.
    const auto factor = std::pow(10.0, decimals >= 0 ? static_cast<double>(decimals) : -static_cast<double>(decimals));
    auto scale = context.mark_node(v0::Constant::create(element::f64, Shape{}, {factor}));
    scale = context.mark_node(std::make_shared<v1::ConvertLike>(scale, x));

    Output<Node> res;
    if (decimals >= 0) {
        const auto scaled = context.mark_node(std::make_shared<v1::Multiply>(x, scale));
        const auto rounded = context.mark_node(std::make_shared<v5::Round>(scaled, v5::Round::RoundMode::HALF_TO_EVEN));
        res = context.mark_node(std::make_shared<v1::Divide>(rounded, scale));
    } else {
        const auto scaled = context.mark_node(std::make_shared<v1::Divide>(x, scale));
        // ConvertDivide rewrites a division into a multiplication by a rounded reciprocal, which
        // moves the pre-round quotient across a tie: f32 6500 / 10^3 is exactly 6.5 and rounds to
        // 6000, while the reciprocal product is 6.5000005 and rounds to 7000.
        disable_divide_conversion(scaled);
        const auto rounded = context.mark_node(std::make_shared<v5::Round>(scaled, v5::Round::RoundMode::HALF_TO_EVEN));
        res = context.mark_node(std::make_shared<v1::Multiply>(rounded, scale));
    }
    // ConvertLike instead of Convert: the input element type may still be dynamic, and it is the
    // one op that resolves the target type when the input is specialized.
    return widen ? context.mark_node(std::make_shared<v1::ConvertLike>(res, data)) : res;
}
}  // namespace

OutputVector translate_round(const NodeContext& context) {
    // aten::round(Tensor self) -> Tensor
    // aten::round(Tensor self, int decimals) -> Tensor                     (TorchScript)
    // aten::round(Tensor self, int decimals, Tensor(a!) out) -> Tensor(a!) (TorchScript)
    // aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    // aten::round.decimals(Tensor self, *, int decimals) -> Tensor         (FX, decimals is a kwarg)
    // aten::round.decimals_out(Tensor self, *, int decimals, Tensor(a!) out) -> Tensor(a!) (FX)
    // aten::round.int(int a) -> float
    // aten::round.float(float a) -> float
    // aten::round.Scalar(Scalar a) -> Scalar
    num_inputs_check(context, 1, 3);
    auto data = context.get_input(0);
    int64_t decimals = 0;
    // Index of the "out" tensor, 0 when this overload has none (index 0 always holds the data).
    size_t out_index = 0;
    if (context.has_attribute("decimals")) {
        // FX path: decimals is a node kwarg, the out kwarg is mutated by the translation session
        decimals = context.get_attribute<int64_t>("decimals");
    } else if (context.get_input_size() > 1 && !context.input_is_none(1) && is_python_scalar_input(context, 1)) {
        // TorchScript path: decimals is an inlined integer scalar in the second position
        decimals = context.const_input<int64_t>(1);
    } else if (context.get_input_size() > 1 && !context.input_is_none(1)) {
        // TorchScript aten::round.out: the second input is the "out" tensor
        out_index = 1;
    }
    if (context.get_input_size() > 2 && !context.input_is_none(2)) {
        // TorchScript aten::round(self, decimals, out): the third input is the "out" tensor
        out_index = 2;
    }
    Output<Node> res;
    if (decimals == 0) {
        res = context.mark_node(std::make_shared<v5::Round>(data, v5::Round::RoundMode::HALF_TO_EVEN));
    } else {
        res = round_with_decimals(context, data, decimals);
    }
    if (out_index != 0) {
        context.mutate_input(out_index, res);
    }
    return {res};
};

}  // namespace ov::frontend::pytorch::op
