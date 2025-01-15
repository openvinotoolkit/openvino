// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_linspace(const NodeContext& context) {
    num_inputs_check(context, 3, 7);
    // "aten::linspace(Scalar start, Scalar end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device?
    // device=None, bool? pin_memory=None) -> Tensor"

    // "aten::linspace.out(Scalar start, Scalar end, int steps, *, Tensor(a!) out) -> Tensor(a!)"
    auto start = context.mark_node(std::make_shared<v0::Convert>(context.get_input(0), element::f32));
    auto end = context.mark_node(std::make_shared<v0::Convert>(context.get_input(1), element::f32));
    auto steps = context.mark_node(std::make_shared<v0::Convert>(context.get_input(2), element::f32));
    auto out_tensor = context.get_input(1);
    auto apply_dtype = true;
    auto dtype = element::f32;
    if (!context.input_is_none(3) && context.get_input_size() == 7) {
        // Case where dtype is provided directly in dtype input.
        if (ov::as_type_ptr<v0::Constant>(context.get_input_from_visible_context(3).get_node_shared_ptr())) {
            dtype = convert_dtype(context.const_input<int64_t>(3));
            apply_dtype = true;
        } else if (const auto& fw_node = cast_fw_node(context.get_input(3).get_node_shared_ptr(), "prim::dtype")) {
            out_tensor = fw_node->input_value(0);
            apply_dtype = false;
        } else {
            PYTORCH_OP_CONVERSION_CHECK(false, "Couldn't get dtype input");
        }
    } else if (!context.input_is_none(3) && context.get_input_size() == 4) {
        // Case where dtype is inherited from out tensor.
        out_tensor = context.get_input(3);
        apply_dtype = false;
    }

    auto const_0 = v0::Constant::create(element::f32, Shape{}, {0});
    auto const_1 = v0::Constant::create(element::f32, Shape{}, {1});
    auto step_range = context.mark_node(std::make_shared<v4::Range>(const_0, steps, const_1, element::f32));

    auto sub_end_start = context.mark_node(std::make_shared<v1::Subtract>(end, start));
    auto sub_steps_1 = context.mark_node(std::make_shared<v1::Subtract>(steps, const_1));
    auto step_multiplier = context.mark_node(std::make_shared<v1::Divide>(sub_end_start, sub_steps_1));
    auto is_single_step = context.mark_node(std::make_shared<v1::Equal>(steps, const_1));
    auto select_multiplier = context.mark_node(std::make_shared<v1::Select>(is_single_step, const_0, step_multiplier));
    auto step_values = context.mark_node(std::make_shared<v1::Multiply>(step_range, select_multiplier));

    auto linspace = context.mark_node(std::make_shared<v1::Add>(step_values, start));
    if (apply_dtype) {
        linspace = context.mark_node(std::make_shared<v0::Convert>(linspace, dtype));
    } else {
        linspace = context.mark_node(std::make_shared<v1::ConvertLike>(linspace, out_tensor));
    }

    return {linspace};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
