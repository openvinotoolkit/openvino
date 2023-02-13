// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/range.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_arange(NodeContext& context) {
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto dtype = element::f32;
    bool dtype_applied = false;
    auto num_inputs = context.get_input_size();
    ov::Output<Node> end;
    ov::Output<Node> out_tensor;
    ov::Output<Node> start = zero;
    ov::Output<Node> step = one;

    if (num_inputs == 2) {
        // aten::arange(Scalar end, tensor out)
        end = context.get_input(0);
        out_tensor = context.input_is_none(1) ? end : context.get_input(1);
    } else if (num_inputs == 4) {
        // aten::arange(Scalar start, Scalar end, Scalar step, Tensor out)
        start = context.get_input(0);
        end = context.get_input(1);
        step = context.get_input(2);
        out_tensor = context.input_is_none(3) ? end : context.get_input(3);
    } else if (num_inputs == 5) {
        // aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
        end = context.get_input(0);
        out_tensor = end;
        if (!context.input_is_none(1)) {
            dtype = convert_dtype(context.const_input<int64_t>(1));
            dtype_applied = true;
        }
    } else if (num_inputs == 6) {
        // aten::arange(Scalar start, Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
        start = context.get_input(0);
        end = context.get_input(1);
        out_tensor = end;
        if (!context.input_is_none(2)) {
            dtype = convert_dtype(context.const_input<int64_t>(2));
            dtype_applied = true;
        }
    } else if (num_inputs == 7) {
        // aten::arange(Scalar start, Scalar end, Scalar step, ScalarType dtype, Layout, Device, bool pin_memory)
        start = context.get_input(0);
        end = context.get_input(1);
        step = context.get_input(2);
        out_tensor = end;
        if (!context.input_is_none(3)) {
            dtype = convert_dtype(context.const_input<int64_t>(3));
            dtype_applied = true;
        }
    } else {
        FRONT_END_OP_CONVERSION_CHECK(false, "Not expected number of inputs for ", context.get_op_type());
    }
    auto r_end = context.mark_node(std::make_shared<v0::Convert>(end, dtype));
    auto r_start = context.mark_node(std::make_shared<v0::Convert>(start, dtype));
    auto r_step = context.mark_node(std::make_shared<v0::Convert>(step, dtype));
    auto range = context.mark_node(std::make_shared<v4::Range>(r_start, r_end, r_step, dtype));
    if (!dtype_applied) {
        range = context.mark_node(std::make_shared<v1::ConvertLike>(range, out_tensor));
    }
    return {range};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov