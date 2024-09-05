// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
Output<Node> base_translate_full(const NodeContext& context, const Output<Node>& sizes, const Output<Node>& value) {
    if (is_empty_list(sizes)) {
        return value;
    }
    return context.mark_node(std::make_shared<v3::Broadcast>(value, sizes));
}

Output<Node> base_translate_full_with_convertlike(const NodeContext& context,
                                                  const Output<Node>& sizes,
                                                  const Output<Node>& value,
                                                  const Output<Node>& out) {
    auto filled_tensor = base_translate_full(context, sizes, value);
    return context.mark_node(std::make_shared<v1::ConvertLike>(filled_tensor, out));
}

Output<Node> base_translate_full_with_convert(const NodeContext& context,
                                              const Output<Node>& sizes,
                                              Output<Node> value,
                                              size_t dtype_id) {
    if (!context.input_is_none(dtype_id)) {
        value = apply_dtype(context, dtype_id, value);
    }

    auto filled_tensor = base_translate_full(context, sizes, value);
    return filled_tensor;
}
}  // namespace

OutputVector translate_full(const NodeContext& context) {
    num_inputs_check(context, 2, 6);
    auto sizes = get_input_concat_if_list(context, 0);
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

OutputVector translate_full_fx(const NodeContext& context) {
    // aten.full.default([16, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'),
    // pin_memory = False)
    auto num_inputs = context.get_input_size();
    num_inputs_check(context, 2, num_inputs);
    ov::Output<ov::Node> sizes;
    if (context.get_input_type(0).is<type::List>()) {
        sizes = concat_list_from_inputs(context, 0, num_inputs - 1);
    } else {
        sizes = context.get_input(0);
    }
    auto value = context.get_input(static_cast<int>(num_inputs - 1));

    auto filled_tensor = base_translate_full(context, sizes, value);
    if (context.has_attribute("dtype")) {
        auto dtype = context.get_attribute<element::Type>("dtype");
        filled_tensor = context.mark_node(std::make_shared<v0::Convert>(filled_tensor, dtype));
    }
    return {filled_tensor};
};

OutputVector translate_full_like(const NodeContext& context) {
    num_inputs_check(context, 2, 7);
    auto input = context.get_input(0);
    auto value = context.get_input(1);
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    if (context.get_input_size() == 7 && !context.input_is_none(2)) {
        return {base_translate_full_with_convert(context, sizes, value, 2)};
    }
    const auto& out = context.input_is_none(3) ? input : context.get_input(3);
    return {base_translate_full_with_convertlike(context, sizes, value, out)};
};

OutputVector translate_full_like_fx(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto input = context.get_input(0);
    auto value = context.get_input(1);
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    auto filled_tensor = base_translate_full(context, sizes, value);
    if (context.has_attribute("dtype")) {
        auto dtype = context.get_attribute<element::Type>("dtype");
        filled_tensor = context.mark_node(std::make_shared<v0::Convert>(filled_tensor, dtype));
    } else {
        filled_tensor = context.mark_node(std::make_shared<v1::ConvertLike>(filled_tensor, input));
    }
    return {filled_tensor};
};

OutputVector translate_fill(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto input = context.get_input(0);
    auto value = context.get_input(1);
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    const auto& out = context.input_is_none(2) ? input : context.get_input(2);
    auto result = base_translate_full_with_convertlike(context, sizes, value, out);
    if (!context.input_is_none(2)) {
        context.mutate_input(2, result);
    }
    return {result};
};

OutputVector translate_new_full(const NodeContext& context) {
    num_inputs_check(context, 3, 7);
    auto input = context.get_input(0);
    auto sizes = get_input_concat_if_list(context, 1);
    auto value = context.get_input(2);
    if (context.get_input_size() == 7 && !context.input_is_none(3)) {
        return {base_translate_full_with_convert(context, sizes, value, 3)};
    }
    return {base_translate_full_with_convertlike(context, sizes, value, input)};
};

OutputVector translate_new_full_fx(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    auto input = context.get_input(0);
    auto sizes = context.get_input(1);
    auto value = context.get_input(2);
    auto filled_tensor = base_translate_full(context, sizes, value);
    if (context.has_attribute("dtype")) {
        auto dtype = context.get_attribute<element::Type>("dtype");
        filled_tensor = context.mark_node(std::make_shared<v0::Convert>(filled_tensor, dtype));
    } else {
        filled_tensor = context.mark_node(std::make_shared<v1::ConvertLike>(filled_tensor, input));
    }
    return {filled_tensor};
};

OutputVector translate_zeros(const NodeContext& context) {
    num_inputs_check(context, 2, 5);
    auto sizes = get_input_concat_if_list(context, 0);
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

OutputVector translate_zeros_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto sizes = context.get_input(0);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    auto filled_tensor = base_translate_full(context, sizes, value);
    if (context.has_attribute("dtype")) {
        auto dtype = context.get_attribute<element::Type>("dtype");
        filled_tensor = context.mark_node(std::make_shared<v0::Convert>(filled_tensor, dtype));
    }
    return {filled_tensor};
};

OutputVector translate_zeros_like(const NodeContext& context) {
    num_inputs_check(context, 1, 6);
    auto input = context.get_input(0);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    if (context.get_input_size() == 6 && !context.input_is_none(1)) {
        return {base_translate_full_with_convert(context, sizes, value, 1)};
    }
    const auto& out = context.input_is_none(2) ? input : context.get_input(2);
    return {base_translate_full_with_convertlike(context, sizes, value, out)};
};

OutputVector translate_zeros_like_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto input = context.get_input(0);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    auto filled_tensor = base_translate_full(context, sizes, value);
    if (context.has_attribute("dtype")) {
        auto dtype = context.get_attribute<element::Type>("dtype");
        filled_tensor = context.mark_node(std::make_shared<v0::Convert>(filled_tensor, dtype));
    } else {
        filled_tensor = context.mark_node(std::make_shared<v1::ConvertLike>(filled_tensor, input));
    }
    return {filled_tensor};
};

OutputVector translate_new_zeros(const NodeContext& context) {
    num_inputs_check(context, 2, 6);
    auto input = context.get_input(0);
    auto sizes = get_input_concat_if_list(context, 1);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    if (context.get_input_size() == 6 && !context.input_is_none(2)) {
        return {base_translate_full_with_convert(context, sizes, value, 2)};
    }
    return {base_translate_full_with_convertlike(context, sizes, value, input)};
};

OutputVector translate_new_zeros_fx(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto input = context.get_input(0);
    auto sizes = context.get_input(1);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    auto filled_tensor = base_translate_full(context, sizes, value);
    if (context.has_attribute("dtype")) {
        auto dtype = context.get_attribute<element::Type>("dtype");
        filled_tensor = context.mark_node(std::make_shared<v0::Convert>(filled_tensor, dtype));
    } else {
        filled_tensor = context.mark_node(std::make_shared<v1::ConvertLike>(filled_tensor, input));
    }
    return {filled_tensor};
};

OutputVector translate_ones(const NodeContext& context) {
    num_inputs_check(context, 1, 5);
    auto sizes = get_input_concat_if_list(context, 0);
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

OutputVector translate_ones_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto sizes = context.get_input(0);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
    auto filled_tensor = base_translate_full(context, sizes, value);
    if (context.has_attribute("dtype")) {
        auto dtype = context.get_attribute<element::Type>("dtype");
        filled_tensor = context.mark_node(std::make_shared<v0::Convert>(filled_tensor, dtype));
    }
    return {filled_tensor};
};

OutputVector translate_ones_like(const NodeContext& context) {
    num_inputs_check(context, 1, 6);
    auto input = context.get_input(0);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    if (context.get_input_size() == 6 && !context.input_is_none(1)) {
        return {base_translate_full_with_convert(context, sizes, value, 1)};
    }
    const auto& out = context.input_is_none(2) ? input : context.get_input(2);
    return {base_translate_full_with_convertlike(context, sizes, value, out)};
};

OutputVector translate_ones_like_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto input = context.get_input(0);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    auto filled_tensor = base_translate_full(context, sizes, value);
    if (context.has_attribute("dtype")) {
        auto dtype = context.get_attribute<element::Type>("dtype");
        filled_tensor = context.mark_node(std::make_shared<v0::Convert>(filled_tensor, dtype));
    } else {
        filled_tensor = context.mark_node(std::make_shared<v1::ConvertLike>(filled_tensor, input));
    }
    return {filled_tensor};
};

OutputVector translate_new_ones(const NodeContext& context) {
    num_inputs_check(context, 2, 6);
    auto input = context.get_input(0);
    auto sizes = get_input_concat_if_list(context, 1);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
    if (context.get_input_size() == 6 && !context.input_is_none(2)) {
        return {base_translate_full_with_convert(context, sizes, value, 2)};
    }
    return {base_translate_full_with_convertlike(context, sizes, value, input)};
};

OutputVector translate_new_ones_fx(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto input = context.get_input(0);
    auto sizes = context.get_input(1);
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1}));
    auto filled_tensor = base_translate_full(context, sizes, value);
    if (context.has_attribute("dtype")) {
        auto dtype = context.get_attribute<element::Type>("dtype");
        filled_tensor = context.mark_node(std::make_shared<v0::Convert>(filled_tensor, dtype));
    } else {
        filled_tensor = context.mark_node(std::make_shared<v1::ConvertLike>(filled_tensor, input));
    }
    return {filled_tensor};
};

OutputVector translate_empty(const NodeContext& context) {
    // aten::empty(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool?
    // pin_memory=None, MemoryFormat? memory_format=None) -> Tensor layout, device and work with memory ignored on our
    // side, so just skip these parameters
    num_inputs_check(context, 1, 6);
    auto sizes = get_input_concat_if_list(context, 0);
    // In OV uninitialized data is not supported, so we create a tensor filled with zeros with a given shape and type.
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

OutputVector translate_empty_like(const NodeContext& context) {
    // aten::empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool?
    // pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
    // aten::empty_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 1, 6);
    auto input = context.get_input(0);
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    // In OV uninitialized data is not supported, so we create a tensor filled with zeros with a given shape and type.
    auto value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
    int dtype_id = 1;
    Output<Node> empty;
    if (context.get_input_size() == 6) {
        if (!context.input_is_none(dtype_id)) {
            empty = base_translate_full_with_convert(context, sizes, value, dtype_id);
        } else {
            empty = base_translate_full_with_convertlike(context, sizes, value, input);
        }
    } else if (context.get_input_size() == 4) {
        const auto& out = context.input_is_none(3) ? input : context.get_input(3);
        empty = base_translate_full_with_convertlike(context, sizes, value, out);
        if (!context.input_is_none(3)) {
            context.mutate_input(3, empty);
        }
    } else {
        FRONT_END_GENERAL_CHECK(false, "Unexpected number of inputs.");
    }
    return {empty};
};

OutputVector translate_fill_diagonal(const NodeContext& context) {
    // aten::fill_diagonal_(Tensor(a!) self, Scalar fill_value, bool wrap=False) -> Tensor(a!)
    // realization inspired by numpy:
    // https://github.com/numpy/numpy/blob/c236e694d222ae6b812cb8dab54471bc4c912f0f/numpy/lib/_index_tricks_impl.py#L787-L918
    num_inputs_check(context, 3, 3);
    auto input_tensor = context.get_input(0);
    auto fill_value = context.get_input(1);
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input_tensor, element::i32));
    auto input_rank = input_tensor.get_partial_shape().rank();
    auto const_one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto const_zero = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto const_one_s = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto const_zero_s = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto const_neg_one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    if (input_rank.is_dynamic() || input_rank.get_length() < 2) {
        PYTORCH_OP_CONVERSION_CHECK(false, "aten::fill_diagonal_ required tensor with static rank >= 2 ");
    }
    auto flatten_input = context.mark_node(std::make_shared<v1::Reshape>(input_tensor, const_neg_one, false));
    auto wrap = context.const_input<bool>(2);
    Output<Node> step;
    // default value for end - number of elements in input tensor
    Output<Node> end;
    auto flatten_shape = context.mark_node(std::make_shared<v3::ShapeOf>(flatten_input, element::i32));
    end = context.mark_node(std::make_shared<v8::Gather>(flatten_shape, const_neg_one, const_zero));
    auto last_dim = context.mark_node(std::make_shared<v8::Gather>(input_shape, const_neg_one, const_zero));
    if (input_rank.get_length() == 2) {
        // step = a.shape[1] + 1
        step = context.mark_node(std::make_shared<v1::Add>(last_dim, const_one_s));
        if (!wrap) {
            // if not wrap. and non squared matrix, do not fill tail by cutting end to square
            end = context.mark_node(std::make_shared<v1::Multiply>(last_dim, last_dim));
        }
    } else {
        // step = 1 + (cumprod(a.shape[:-1])).sum()
        // cumprod operation is not supported by ov, but with condition that >2D tensors supported only if all dims
        // equals cumprod can be represented as finite geometric serial and its sum can be found by formula
        // b0 * (bn * q - 1) / (q - 1), where in this particual case q = b0, bn = b0 ^ n
        auto rank_minus_one =
            context.mark_node(v0::Constant::create(element::i32, Shape{}, {input_rank.get_length() - 1}));
        auto dim_power = context.mark_node(std::make_shared<v1::Power>(last_dim, rank_minus_one));
        auto dim_power_minus_one = context.mark_node(std::make_shared<v1::Add>(dim_power, const_neg_one));
        auto dim_minus_one = context.mark_node(std::make_shared<v1::Add>(last_dim, const_neg_one));
        auto q = context.mark_node(std::make_shared<v1::Divide>(dim_power_minus_one, dim_minus_one, true));
        auto cumprod_sum = context.mark_node(std::make_shared<v1::Multiply>(last_dim, q));
        step = context.mark_node(std::make_shared<v1::Add>(const_one_s, cumprod_sum));
        // wrap parameter is not applicable in this case as supported only equal dims on pytorch side
    }
    step = context.mark_node(std::make_shared<v0::Squeeze>(step, const_zero));
    end = context.mark_node(std::make_shared<v0::Squeeze>(end, const_zero));
    auto indices = context.mark_node(std::make_shared<v4::Range>(const_zero_s, end, step, element::i32));
    auto indices_shape = context.mark_node(std::make_shared<v3::ShapeOf>(indices, element::i32));
    fill_value = context.mark_node(std::make_shared<v1::ConvertLike>(fill_value, input_tensor));
    fill_value = context.mark_node(std::make_shared<v1::Broadcast>(fill_value, indices_shape));
    // fill values
    auto filled_tensor =
        context.mark_node(std::make_shared<v12::ScatterElementsUpdate>(flatten_input, indices, fill_value, const_zero));
    // reshape back to original shape
    filled_tensor = context.mark_node(std::make_shared<v1::Reshape>(filled_tensor, input_shape, false));
    return {filled_tensor};
}
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
