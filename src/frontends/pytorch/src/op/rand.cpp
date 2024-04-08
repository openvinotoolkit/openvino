// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <random>

#include "openvino/frontend/common/random_normal_helper.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sqrt.hpp"
#include "pt_framework_node.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
OutputVector make_random_normal(const NodeContext& context,
                                const Output<Node>& sizes,
                                element::Type target_type,
                                const Output<Node>& scale_const,
                                const Output<Node>& mean_const) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(0.0f, 9999.0f);
    float seed = distrib(gen);

    pass::NodeRegistry registry;
    auto res = ov::frontend::make_random_normal(registry, sizes, target_type, mean_const, scale_const, seed);
    context.mark_nodes(registry.get());
    return res;
}
};  // namespace

OutputVector translate_rand(const NodeContext& context) {
    num_inputs_check(context, 1, 6);
    auto sizes = context.get_input(0);
    if (context.get_input_type(0).is<type::List>()) {
        sizes = concat_list_construct(sizes);
    }
    sizes = context.mark_node(std::make_shared<v0::Convert>(sizes, element::i32));
    auto low = context.mark_node(v0::Constant::create(element::f32, Shape{1}, {0}));
    auto high = context.mark_node(v0::Constant::create(element::f32, Shape{1}, {1}));
    auto dtype = element::f32;
    size_t out_id = 1;
    if (context.get_input_size() == 3) {
        PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(1),
                                    "aten::rand conversion with generator does not supported");
        out_id = 2;
    }
    // aten::rand.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
    // aten::rand.generator_out(SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
    if (context.get_input_size() <= 3) {
        auto res = context.mark_node(std::make_shared<v8::RandomUniform>(sizes, low, high, dtype));
        if (context.get_input_size() >= 2)
            context.mutate_input(out_id, res);
        return {res};
    }
    // aten::rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool?
    // pin_memory=None) -> Tensor
    // aten::rand.generator(SymInt[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None,
    // Device? device=None, bool? pin_memory=None) -> Tensor
    bool dtype_applied = true;
    Output<Node> convert_like_out;
    size_t dtype_id = 1;
    if (context.get_input_size() == 6) {
        PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(1),
                                    "aten::rand conversion with generator does not supported");
        dtype_id = 2;
    }
    if (!context.input_is_none(dtype_id)) {
        if (std::dynamic_pointer_cast<v0::Constant>(
                context.get_input_from_visible_context(dtype_id).get_node_shared_ptr())) {
            dtype = convert_dtype(context.const_input<int64_t>(dtype_id));
            low = context.mark_node(std::make_shared<v0::Convert>(low, dtype));
            high = context.mark_node(std::make_shared<v0::Convert>(high, dtype));
        } else if (const auto& fw_node =
                       cast_fw_node(context.get_input(static_cast<int>(dtype_id)).get_node_shared_ptr(),
                                    "prim::dtype")) {
            convert_like_out = fw_node->input_value(0);
            dtype_applied = false;

        } else {
            PYTORCH_OP_CONVERSION_CHECK(false, "Couldn't get dtype input");
        }
    }
    auto res = context.mark_node(std::make_shared<v8::RandomUniform>(sizes, low, high, dtype));
    if (!dtype_applied) {
        res = context.mark_node(std::make_shared<v1::ConvertLike>(res, convert_like_out));
    }
    return {res};
};

OutputVector translate_rand_like(const NodeContext& context) {
    // aten::rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool?
    // pin_memory=None, MemoryFormat? memory_format=None) -> Tensor aten::rand_like.out(Tensor self, *,
    // MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 3, 6);
    auto inp_tensor = context.get_input(0);
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(inp_tensor, element::i32));
    auto low = context.mark_node(v0::Constant::create(element::f32, Shape{1}, {0}));
    auto high = context.mark_node(v0::Constant::create(element::f32, Shape{1}, {1}));
    auto dtype = element::f32;
    if (context.get_input_size() == 3) {
        auto res = context.mark_node(std::make_shared<v8::RandomUniform>(sizes, low, high, dtype));
        context.mutate_input(2, res);
        return {res};
    }
    bool dtype_applied = true;
    Output<Node> convert_like_out;
    if (!context.input_is_none(1)) {
        if (std::dynamic_pointer_cast<v0::Constant>(context.get_input_from_visible_context(1).get_node_shared_ptr())) {
            dtype = convert_dtype(context.const_input<int64_t>(1));
            low = context.mark_node(std::make_shared<v0::Convert>(low, dtype));
            high = context.mark_node(std::make_shared<v0::Convert>(high, dtype));
        } else if (const auto& fw_node =
                       cast_fw_node(context.get_input(static_cast<int>(1)).get_node_shared_ptr(), "prim::dtype")) {
            convert_like_out = fw_node->input_value(0);
            dtype_applied = false;

        } else {
            PYTORCH_OP_CONVERSION_CHECK(false, "Couldn't get dtype input");
        }
    }
    auto res = context.mark_node(std::make_shared<v8::RandomUniform>(sizes, low, high, dtype));
    if (!dtype_applied) {
        res = context.mark_node(std::make_shared<v1::ConvertLike>(res, convert_like_out));
    }
    return {res};
};

OutputVector translate_randn(const NodeContext& context) {
    num_inputs_check(context, 2, 6);
    auto sizes = context.get_input(0);
    if (context.get_input_type(0).is<type::List>()) {
        sizes = concat_list_construct(sizes);
    }
    sizes = context.mark_node(std::make_shared<v0::Convert>(sizes, element::i32));
    auto dtype = element::f32;
    size_t out_id = 1;
    if (context.get_input_size() == 3) {
        PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(1),
                                    "aten::randn conversion with generator does not supported");
        out_id = 2;
    }
    // aten::randn.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
    // aten::randn.generator_out(SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
    if (context.get_input_size() == 2 || context.get_input_size() == 3) {
        auto scale = context.mark_node(v0::Constant::create(dtype, Shape{1}, {1}));
        auto mean = context.mark_node(v0::Constant::create(dtype, Shape{1}, {0}));
        auto res = make_random_normal(context, sizes, dtype, scale, mean);
        context.mutate_input(out_id, res[0]);
        return res;
    }
    size_t dtype_id = 1;
    if (context.get_input_size() == 6) {
        PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(1),
                                    "aten::randn conversion with generator does not supported");
        dtype_id = 2;
    }
    // aten::randn(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool?
    // pin_memory=None) -> Tensor
    // aten::randn.generator(SymInt[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None,
    // Device? device=None, bool? pin_memory=None) -> Tensor
    bool dtype_applied = true;
    Output<Node> convert_like_out;
    if (!context.input_is_none(dtype_id)) {
        if (std::dynamic_pointer_cast<v0::Constant>(
                context.get_input_from_visible_context(dtype_id).get_node_shared_ptr())) {
            dtype = convert_dtype(context.const_input<int64_t>(dtype_id));
        } else if (const auto& fw_node =
                       cast_fw_node(context.get_input(static_cast<int>(dtype_id)).get_node_shared_ptr(),
                                    "prim::dtype")) {
            convert_like_out = fw_node->input_value(0);
            dtype_applied = false;

        } else {
            PYTORCH_OP_CONVERSION_CHECK(false, "Couldn't get dtype input");
        }
    }
    auto scale = context.mark_node(v0::Constant::create(dtype, Shape{1}, {1}));
    auto mean = context.mark_node(v0::Constant::create(dtype, Shape{1}, {0}));
    auto res = make_random_normal(context, sizes, dtype, scale, mean);
    if (!dtype_applied) {
        res[0] = context.mark_node(std::make_shared<v1::ConvertLike>(res[0], convert_like_out));
    }
    return res;
};

OutputVector translate_randn_like(const NodeContext& context) {
    // aten::randn_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool?
    // pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
    // aten::rand_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 3, 6);
    auto inp_tensor = context.get_input(0);
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(inp_tensor, element::i32));
    auto dtype = element::f32;
    if (context.get_input_size() == 3) {
        auto scale = context.mark_node(v0::Constant::create(dtype, Shape{1}, {1}));
        auto mean = context.mark_node(v0::Constant::create(dtype, Shape{1}, {0}));
        auto res = make_random_normal(context, sizes, dtype, scale, mean);
        context.mutate_input(2, res[0]);
        return res;
    }
    // aten::rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool?
    // pin_memory=None) -> Tensor
    bool dtype_applied = true;
    Output<Node> convert_like_out;
    if (!context.input_is_none(1)) {
        if (std::dynamic_pointer_cast<v0::Constant>(context.get_input_from_visible_context(1).get_node_shared_ptr())) {
            dtype = convert_dtype(context.const_input<int64_t>(1));
        } else if (const auto& fw_node =
                       cast_fw_node(context.get_input(static_cast<int>(1)).get_node_shared_ptr(), "prim::dtype")) {
            convert_like_out = fw_node->input_value(0);
            dtype_applied = false;

        } else {
            PYTORCH_OP_CONVERSION_CHECK(false, "Couldn't get dtype input");
        }
    }
    auto scale = context.mark_node(v0::Constant::create(dtype, Shape{1}, {1}));
    auto mean = context.mark_node(v0::Constant::create(dtype, Shape{1}, {0}));
    auto res = make_random_normal(context, sizes, dtype, scale, mean);
    if (!dtype_applied) {
        res[0] = context.mark_node(std::make_shared<v1::ConvertLike>(res[0], convert_like_out));
    }
    return res;
};

OutputVector translate_randint(const NodeContext& context) {
    // aten::randint.low(int low, int high, SymInt[] size, *, ScalarType? dtype=4, Layout? layout=None, Device?
    // device=None, bool? pin_memory=None) -> Tensor
    num_inputs_check(context, 7, 7);
    auto low = context.get_input(0);
    auto high = context.get_input(1);
    auto sizes = context.get_input(2);
    auto dtype = element::i64;
    bool dtype_applied = true;
    Output<Node> convert_like_out;
    if (!context.input_is_none(3)) {
        if (std::dynamic_pointer_cast<v0::Constant>(context.get_input_from_visible_context(3).get_node_shared_ptr())) {
            dtype = convert_dtype(context.const_input<int64_t>(3));
        } else if (const auto& fw_node =
                       cast_fw_node(context.get_input(static_cast<int>(3)).get_node_shared_ptr(), "prim::dtype")) {
            convert_like_out = fw_node->input_value(0);
            dtype_applied = false;
        } else {
            PYTORCH_OP_CONVERSION_CHECK(false, "Couldn't get dtype input");
        }
    }
    low = context.mark_node(std::make_shared<v0::Convert>(low, dtype));
    high = context.mark_node(std::make_shared<v0::Convert>(high, dtype));
    auto res = context.mark_node(std::make_shared<v8::RandomUniform>(sizes, low, high, dtype));
    if (!dtype_applied) {
        res = context.mark_node(std::make_shared<v1::ConvertLike>(res, convert_like_out));
    }
    return {res};
};

OutputVector translate_normal_(const NodeContext& context) {
    // aten::normal_(Tensor(a!) self, float mean=0., float std=1., *, Generator? generator=None) -> Tensor(a!)
    num_inputs_check(context, 3, 4);
    auto inp_tensor = context.get_input(0);
    auto mean = context.get_input(1);
    auto std = context.get_input(2);
    auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(inp_tensor, element::i32));
    auto dtype = element::f32;
    auto res = make_random_normal(context, sizes, dtype, std, mean);
    res[0] = context.mark_node(std::make_shared<v1::ConvertLike>(res[0], inp_tensor));
    context.mutate_input(0, res[0]);
    return res;
}

OutputVector translate_normal(const NodeContext& context) {
    num_inputs_check(context, 2, 8);
    auto mean = context.get_input(0);
    auto std = context.get_input(1);
    auto dtype = element::f32;
    if (context.get_input_size() == 3 || context.get_input_size() == 4) {
        // aten::normal.Tensor_float(Tensor mean, float std=1., *, Generator? generator=None) -> Tensor
        // aten::normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator=None) -> Tensor
        // aten::normal.Tensor_float_out(Tensor mean, float std=1., *, Generator? generator=None, Tensor(a!) out) ->
        // Tensor(a!)
        // aten::normal.Tensor_float_out(Tensor mean, float std=1., *, Generator? generator=None, Tensor(a!)
        // out) -> Tensor(a!)
        // aten::normal.Tensor_Tensor_out(Tensor mean, Tensor std, *, Generator? generator=None,
        // Tensor(a!) out) -> Tensor(a!)
        auto sizes = context.mark_node(std::make_shared<v3::ShapeOf>(mean, element::i32));
        auto res = make_random_normal(context, sizes, dtype, std, mean);
        if (!context.input_is_none(3)) {
            // out
            auto out = context.get_input(3);
            res[0] = context.mark_node(std::make_shared<v1::ConvertLike>(res[0], out));
            context.mutate_input(3, res[0]);
        }
        return res;
    } else if (context.get_input_size() == 5) {
        // aten::normal.float_float_out(float mean, float std, SymInt[] size, *, Generator? generator=None, Tensor(a!)
        // out) -> Tensor(a!)
        auto sizes = context.get_input(2);
        auto res = make_random_normal(context, sizes, dtype, std, mean);
        if (!context.input_is_none(4)) {
            // out
            auto out = context.get_input(4);
            res[0] = context.mark_node(std::make_shared<v1::ConvertLike>(res[0], out));
            context.mutate_input(4, res[0]);
        }
        return res;
    } else if (context.get_input_size() == 8) {
        // aten::normal.float_float(float mean, float std, SymInt[] size, *, Generator? generator=None, ScalarType?
        // dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        auto sizes = context.get_input(2);
        Output<Node> convert_like_out;
        bool dtype_applied = true;
        if (!context.input_is_none(4)) {
            if (std::dynamic_pointer_cast<v0::Constant>(
                    context.get_input_from_visible_context(3).get_node_shared_ptr())) {
                dtype = convert_dtype(context.const_input<int64_t>(4));
            } else if (const auto& fw_node = cast_fw_node(context.get_input(3).get_node_shared_ptr(), "prim::dtype")) {
                convert_like_out = fw_node->input_value(0);
                dtype_applied = false;
            } else {
                PYTORCH_OP_CONVERSION_CHECK(false, "Couldn't get dtype input");
            }
        }
        auto res = make_random_normal(context, sizes, dtype, std, mean);
        if (!dtype_applied) {
            res[0] = context.mark_node(std::make_shared<v1::ConvertLike>(res[0], convert_like_out));
        }
        return res;
    } else {
        PYTORCH_OP_CONVERSION_CHECK(false,
                                    "Unsupported number of inputs to aten::normal operation: ",
                                    context.get_input_size());
    }
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
