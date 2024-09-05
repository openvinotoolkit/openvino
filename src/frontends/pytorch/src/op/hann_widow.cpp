// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES
#include <math.h>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_hann_window(const NodeContext& context) {
    // aten::hann_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool?
    // pin_memory=None) -> Tensor
    // aten::hann_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None,
    // Device? device=None, bool? pin_memory=None) -> Tensor
    // aten::hann_window.out(int window_length, *, Tensor(a!) out) -> Tensor(a!)
    // aten::hann_window.periodic_out(int window_length, bool periodic, *, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 1, 6);
    auto window_size = context.get_input(0);
    bool periodic = true;
    auto num_inputs = context.get_input_size();
    if ((num_inputs == 3 || num_inputs == 6) && !context.input_is_none(1)) {
        periodic = context.const_input<bool>(1);
    }
    auto zero_f = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    auto one_f = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
    auto two_f = context.mark_node(v0::Constant::create(element::f32, Shape{}, {2}));
    auto window_size_f = context.mark_node(std::make_shared<v0::Convert>(window_size, element::f32));
    auto range = context.mark_node(std::make_shared<v4::Range>(zero_f, window_size_f, one_f, ov::element::f32));
    auto pi = context.mark_node(v0::Constant::create(ov::element::f32, Shape{}, {static_cast<float>(M_PI)}));
    auto output = context.mark_node(std::make_shared<v1::Multiply>(range, pi));
    auto factor = window_size_f;
    if (!periodic) {
        factor = context.mark_node(std::make_shared<v1::Subtract>(window_size_f, one_f));
    }
    output = context.mark_node(std::make_shared<v1::Divide>(output, factor));
    auto sin = context.mark_node(std::make_shared<v0::Sin>(output));
    Output<Node> squared_sin = context.mark_node(std::make_shared<v1::Power>(sin, two_f));
    if (num_inputs > 3) {
        size_t dtype_id = num_inputs == 5 ? 1 : 2;
        if (!context.input_is_none(dtype_id)) {
            squared_sin = apply_dtype(context, dtype_id, squared_sin);
        }
    } else {
        size_t out_id = num_inputs == 3 ? 2 : 1;
        if (!context.input_is_none(out_id)) {
            squared_sin = context.mark_node(
                std::make_shared<v1::ConvertLike>(squared_sin, context.get_input(static_cast<int>(out_id))));
            context.mutate_input(out_id, squared_sin);
        }
    }
    return {squared_sin};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov