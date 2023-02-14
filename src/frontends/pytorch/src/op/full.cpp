// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
Output<Node> base_translate_full(const NodeContext& context, const Output<Node>& sizes, const Output<Node>& value) {
    return context.mark_node(std::make_shared<v3::Broadcast>(value, sizes));
}

Output<Node> base_translate_full_with_convert(const NodeContext& context,
                                              const Output<Node>& sizes,
                                              const Output<Node>& value,
                                              size_t dtype_id) {
    auto filled_tensor = base_translate_full(context, sizes, value);
    if (!context.input_is_none(dtype_id)) {
        auto dtype = convert_dtype(context.const_input<int64_t>(dtype_id));
        filled_tensor = context.mark_node(std::make_shared<v0::Convert>(filled_tensor, dtype));
    }
    return filled_tensor;
}

Output<Node> base_translate_full_with_convertlike(const NodeContext& context,
                                                  const Output<Node>& sizes,
                                                  const Output<Node>& value,
                                                  const Output<Node>& out) {
    auto filled_tensor = base_translate_full(context, sizes, value);
    return context.mark_node(std::make_shared<v1::ConvertLike>(filled_tensor, out));
}
}  // namespace

OutputVector translate_full(NodeContext& context) {
    num_inputs_check(context, 2, 6);
    auto sizes = context.get_input(0);
    auto value = context.get_input(1);
    auto num_inputs = context.get_input_size();
    if (num_inputs < 6) {
        int out_id = num_inputs == 3 ? 2 : 3;
        if (!context.input_is_none(static_cast<size_t>(out_id))) {
            auto out = context.get_input(out_id);
            return {base_translate_full_with_convertlike(context, sizes, value, out)};
        }
        return {base_translate_full(context, sizes, value)};
    }
    size_t dtype_id = num_inputs == 6 ? 2 : 3;
    return {base_translate_full_with_convert(context, sizes, value, dtype_id)};
};

OutputVector translate_full_like(NodeContext& context) {
    num_inputs_check(context, 2, 7);
    auto input = context.get_input(0);
    auto value = context.get_input(1);
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(input));
    if (context.get_input_size() == 7) {
        return {base_translate_full_with_convert(context, sizes, value, 2)};
    }
    auto out = context.input_is_none(3) ? input : context.get_input(3);
    return {base_translate_full_with_convertlike(context, sizes, value, out)};
};

OutputVector translate_fill_(NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto input = context.get_input(0);
    auto value = context.get_input(1);
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(input));
    return {base_translate_full_with_convertlike(context, sizes, value, input)};
};

OutputVector translate_new_full(NodeContext& context) {
    num_inputs_check(context, 3, 7);
    auto input = context.get_input(0);
    auto sizes = context.get_input(1);
    auto value = context.get_input(2);
    if (context.get_input_size() == 7 && !context.input_is_none(3)) {
        return {base_translate_full_with_convert(context, sizes, value, 3)};
    }
    return {base_translate_full_with_convertlike(context, sizes, value, input)};
};

OutputVector translate_zeros(NodeContext& context) {
    num_inputs_check(context, 2, 5);
    auto sizes = context.get_input(0);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    auto num_inputs = context.get_input_size();
    if (num_inputs < 5) {
        int out_id = num_inputs == 2 ? 1 : 2;
        if (!context.input_is_none(static_cast<size_t>(out_id))) {
            auto out = context.get_input(out_id);
            return {base_translate_full_with_convertlike(context, sizes, value, out)};
        }
        return {base_translate_full(context, sizes, value)};
    }
    size_t dtype_id = num_inputs == 5 ? 1 : 2;
    return {base_translate_full_with_convert(context, sizes, value, dtype_id)};
};

OutputVector translate_zeros_like(NodeContext& context) {
    num_inputs_check(context, 1, 6);
    auto input = context.get_input(0);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(input));
    if (context.get_input_size() == 6) {
        return {base_translate_full_with_convert(context, sizes, value, 1)};
    }
    auto out = context.input_is_none(2) ? input : context.get_input(2);
    return {base_translate_full_with_convertlike(context, sizes, value, out)};
};

OutputVector translate_new_zeros(NodeContext& context) {
    num_inputs_check(context, 2, 6);
    auto input = context.get_input(0);
    auto sizes = context.get_input(1);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    if (context.get_input_size() == 6 && !context.input_is_none(2)) {
        return {base_translate_full_with_convert(context, sizes, value, 2)};
    }
    return {base_translate_full_with_convertlike(context, sizes, value, input)};
};

OutputVector translate_ones(NodeContext& context) {
    num_inputs_check(context, 1, 5);
    auto sizes = context.get_input(0);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
    auto num_inputs = context.get_input_size();
    if (num_inputs < 5) {
        int out_id = num_inputs == 2 ? 1 : 2;
        if (!context.input_is_none(static_cast<size_t>(out_id))) {
            auto out = context.get_input(out_id);
            return {base_translate_full_with_convertlike(context, sizes, value, out)};
        }
        return {base_translate_full(context, sizes, value)};
    }
    size_t dtype_id = num_inputs == 5 ? 1 : 2;
    return {base_translate_full_with_convert(context, sizes, value, dtype_id)};
};

OutputVector translate_ones_like(NodeContext& context) {
    num_inputs_check(context, 1, 6);
    auto input = context.get_input(0);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(input));
    if (context.get_input_size() == 6) {
        return {base_translate_full_with_convert(context, sizes, value, 1)};
    }
    auto out = context.input_is_none(2) ? input : context.get_input(2);
    return {base_translate_full_with_convertlike(context, sizes, value, out)};
};

OutputVector translate_new_ones(NodeContext& context) {
    num_inputs_check(context, 2, 6);
    auto input = context.get_input(0);
    auto sizes = context.get_input(1);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
    if (context.get_input_size() == 6 && !context.input_is_none(2)) {
        return {base_translate_full_with_convert(context, sizes, value, 2)};
    }
    return {base_translate_full_with_convertlike(context, sizes, value, input)};
};

OutputVector translate_empty(NodeContext& context) {
    num_inputs_check(context, 1, 2);
    auto sizes = context.get_input(0);
    // In OV uninitialised data is not supported, so we create a tensor filled with zeros with a given shape and type.
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    int dtype_id = 1;
    Output<Node> empty;
    if (!context.input_is_none(dtype_id)) {
        empty = base_translate_full_with_convert(context, sizes, value, dtype_id);
    } else {
        empty = base_translate_full(context, sizes, value);
    }
    return {empty};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov