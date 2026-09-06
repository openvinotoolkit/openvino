// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/round.hpp"

#include <cmath>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
// torch.round(x, decimals=d) == round_half_to_even(x * 10^d) / 10^d ; d may be negative.
// The plain v5::Round only rounds to integers, so a non-zero decimals value is emulated by
// scaling the input before rounding and scaling it back after. Rounding mode is unchanged
// (half-to-even), so midpoints and the result sign (incl. -0) match torch.
Output<Node> round_with_decimals(const NodeContext& context, const Output<Node>& data, int64_t decimals) {
    // Build the scale in fp64 and ConvertLike it to `data` (same idiom as clamp.cpp): the input
    // element type may still be dynamic during conversion, so it cannot be used to type a constant.
    auto scale = context.mark_node(
        v0::Constant::create(element::f64, Shape{}, {std::pow(10.0, static_cast<double>(decimals))}));
    scale = context.mark_node(std::make_shared<v1::ConvertLike>(scale, data));
    auto scaled = context.mark_node(std::make_shared<v1::Multiply>(data, scale));
    auto rounded = context.mark_node(std::make_shared<v5::Round>(scaled, v5::Round::RoundMode::HALF_TO_EVEN));
    return context.mark_node(std::make_shared<v1::Divide>(rounded, scale));
}
}  // namespace

OutputVector translate_round(const NodeContext& context) {
    // aten::round(Tensor self) -> Tensor
    // aten::round(Tensor self, int decimals) -> Tensor          (TorchScript: decimals is a scalar input 1)
    // aten::round.decimals(Tensor self, int decimals) -> Tensor (FX: decimals is kept in node kwargs)
    // aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    // aten::round.int(int a) -> float
    // aten::round.float(float a) -> float
    // aten::round.Scalar(Scalar a) -> Scalar
    num_inputs_check(context, 1, 2);
    auto data = context.get_input(0);
    int64_t decimals = 0;
    bool out_is_second = false;
    if (context.has_attribute("decimals")) {
        // FX path: aten::round.decimals passes decimals as a node attribute
        decimals = context.get_attribute<int64_t>("decimals");
    } else if (context.get_input_size() > 1 && !context.input_is_none(1) &&
               is_python_scalar_input(context, 1)) {
        // TorchScript path: aten::round(self, decimals) where decimals is an inlined integer scalar
        decimals = context.const_input<int64_t>(1);
    } else if (context.get_input_size() > 1 && !context.input_is_none(1)) {
        // The second argument is the "out" tensor of aten::round.out
        out_is_second = true;
    }
    Output<Node> res;
    if (decimals == 0) {
        res = context.mark_node(std::make_shared<v5::Round>(data, v5::Round::RoundMode::HALF_TO_EVEN));
    } else {
        res = round_with_decimals(context, data, decimals);
    }
    if (out_is_second) {
        context.mutate_input(1, res);
    }
    return {res};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
