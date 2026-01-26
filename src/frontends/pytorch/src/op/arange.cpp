// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/squeeze.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
OutputVector make_arange(const NodeContext& context,
                         const Output<Node>& start,
                         const Output<Node>& end,
                         const Output<Node>& step,
                         const element::Type& dtype,
                         const Output<Node>& ref_tensor = {}) {
    auto range_dtype = element::i64;
    auto is_real_tensor = [](const auto& o) {
        return o.get_element_type().is_real();
    };
    auto is_dynamic_tensor = [](const auto& o) {
        return o.get_element_type().is_dynamic();
    };
    if ((dtype.is_dynamic() || dtype.is_real()) &&
        (is_real_tensor(start) || is_real_tensor(end) || is_real_tensor(step))) {
        range_dtype = element::f32;
    }
    if (is_dynamic_tensor(start) || is_dynamic_tensor(end) || is_dynamic_tensor(step)) {
        // use f32 in dynamic case
        range_dtype = element::f32;
    }
    if (ref_tensor.get_node_shared_ptr() && ref_tensor.get_element_type().is_static() &&
        ref_tensor.get_element_type().is_integral()) {
        // target type is integer, so use int Range
        range_dtype = element::i64;
    }
    auto range = context.mark_node(std::make_shared<v4::Range>(start, end, step, range_dtype));
    if (ref_tensor.get_node_shared_ptr()) {
        range = context.mark_node(std::make_shared<v1::ConvertLike>(range, ref_tensor));
    } else if (dtype != range_dtype && dtype.is_static()) {
        range = context.mark_node(std::make_shared<v0::Convert>(range, dtype));
    }
    return {std::move(range)};
}
}  // namespace

OutputVector translate_arange(const NodeContext& context) {
    const auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    const auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    int dtype_port = -1;
    auto dtype = element::dynamic;
    const auto num_inputs = context.get_input_size();
    ov::Output<Node> end;
    ov::Output<Node> out_tensor;
    ov::Output<Node> start = zero;
    ov::Output<Node> step = one;

    if (num_inputs == 2) {
        // aten::arange(Scalar end, tensor out)
        end = context.get_input(0);
        if (!context.input_is_none(1)) {
            out_tensor = context.get_input(1);
        }
    } else if (num_inputs == 4) {
        // aten::arange(Scalar start, Scalar end, Scalar step, Tensor out)
        start = context.get_input(0);
        end = context.get_input(1);
        step = context.get_input(2);
        if (!context.input_is_none(3)) {
            out_tensor = context.get_input(3);
        }
    } else if (num_inputs == 5) {
        // aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
        end = context.get_input(0);
        if (!context.input_is_none(1)) {
            dtype_port = 1;
        }
    } else if (num_inputs == 6) {
        // aten::arange(Scalar start, Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
        start = context.get_input(0);
        end = context.get_input(1);
        dtype_port = 2;
    } else if (num_inputs == 7) {
        // aten::arange(Scalar start, Scalar end, Scalar step, ScalarType dtype, Layout, Device, bool pin_memory)
        start = context.get_input(0);
        end = context.get_input(1);
        step = context.get_input(2);
        dtype_port = 3;
    } else {
        PYTORCH_OP_CONVERSION_CHECK(false, "Not expected number of inputs for ", context.get_op_type());
    }
    if (!out_tensor.get_node_shared_ptr() && dtype_port >= 0) {
        if (context.input_is_none(dtype_port)) {
            // dtype=None and not provided behave differently, so we rely on output type if we can
            const auto dtype_val = simplified_type_interpret(context.get_output_type(0));
            if (dtype_val.is<element::Type>()) {
                dtype = dtype_val.as<element::Type>();
            }
            if (dtype.is_dynamic()) {
                // convert to default torch dtype (see torch.get_default_dtype())
                dtype = element::f32;
            }
        } else {
            if (ov::as_type_ptr<v0::Constant>(
                    context.get_input_from_visible_context(dtype_port).get_node_shared_ptr())) {
                dtype = convert_dtype(context.const_input<int64_t>(dtype_port));
            } else if (const auto& fw_node =
                           cast_fw_node(context.get_input(dtype_port).get_node_shared_ptr(), "prim::dtype")) {
                out_tensor = fw_node->input_value(0);
            } else {
                PYTORCH_OP_CONVERSION_CHECK(false, "Couldn't get dtype input");
            }
        }
    }
    return make_arange(context, start, end, step, dtype, out_tensor);
}

OutputVector translate_arange_fx(const NodeContext& context) {
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto dtype = element::f32;
    auto num_inputs = context.get_input_size();
    ov::Output<Node> end;
    ov::Output<Node> start = zero;
    ov::Output<Node> step = one;

    if (num_inputs == 1) {
        // arange = torch.ops.aten.arange.default(_local_scalar_dense, dtype = torch.int8, device =
        // device(type='cpu'), pin_memory = False);
        end = context.get_input(0);
    } else if (num_inputs == 2) {
        start = context.get_input(0);
        end = context.get_input(1);
    } else if (num_inputs == 3) {
        start = context.get_input(0);
        end = context.get_input(1);
        step = context.get_input(2);
    } else {
        PYTORCH_OP_CONVERSION_CHECK(false, "Not expected number of inputs for ", context.get_op_type());
    }
    if (context.has_attribute("dtype")) {
        dtype = context.get_attribute<element::Type>("dtype");
    }
    auto input_squeeze = [&context](ov::Output<Node> input) {
        if (input.get_partial_shape().rank().is_dynamic() ||
            (input.get_partial_shape().rank().is_static() && input.get_partial_shape().rank().get_length() == 1)) {
            auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
            input = context.mark_node(std::make_shared<ov::op::v0::Squeeze>(input, zero));
        }
        return input;
    };
    start = input_squeeze(start);
    end = input_squeeze(end);
    step = input_squeeze(step);
    auto range = context.mark_node(std::make_shared<v4::Range>(start, end, step, dtype));
    if (!context.has_attribute("dtype")) {
        range = context.mark_node(std::make_shared<v1::ConvertLike>(range, context.get_input(0)));
    }
    return {range};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
