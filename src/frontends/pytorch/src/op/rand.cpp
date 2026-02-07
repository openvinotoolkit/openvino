// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <random>

#include "openvino/frontend/common/random_normal_helper.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "pt_framework_node.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
constexpr int64_t standard_gamma_trials = 16;
constexpr float min_uniform_value = 1e-7f;
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
        if (ov::as_type_ptr<v0::Constant>(context.get_input_from_visible_context(dtype_id).get_node_shared_ptr())) {
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
        if (ov::as_type_ptr<v0::Constant>(context.get_input_from_visible_context(1).get_node_shared_ptr())) {
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
        if (ov::as_type_ptr<v0::Constant>(context.get_input_from_visible_context(dtype_id).get_node_shared_ptr())) {
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
        if (ov::as_type_ptr<v0::Constant>(context.get_input_from_visible_context(1).get_node_shared_ptr())) {
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
        if (ov::as_type_ptr<v0::Constant>(context.get_input_from_visible_context(3).get_node_shared_ptr())) {
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

OutputVector translate_standard_gamma(const NodeContext& context) {
    // aten::_standard_gamma(Tensor self, *, Generator? generator=None) -> Tensor
    num_inputs_check(context, 1, 2);
    if (context.get_input_size() == 2) {
        PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(1),
                                    "aten::_standard_gamma conversion with generator is not supported");
    }

    auto input = context.get_input(0);
    auto output_type = input.get_element_type();
    auto concentration = input;
    if (output_type != element::f32) {
        concentration = context.mark_node(std::make_shared<v0::Convert>(input, element::f32));
    }

    auto shape_i32 = context.mark_node(std::make_shared<v3::ShapeOf>(concentration, element::i32));
    auto shape = context.mark_node(std::make_shared<v0::Convert>(shape_i32, element::i64));
    auto trials =
        context.mark_node(v0::Constant::create(element::i64, Shape{1}, {standard_gamma_trials}));
    auto expanded_shape =
        context.mark_node(std::make_shared<v0::Concat>(OutputVector{trials, shape}, 0));
    auto axis_zero_i64 = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto axis_zero_i32 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));

    auto zero = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0.f}));
    auto one = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1.f}));
    auto half = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0.5f}));
    auto third = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1.f / 3.f}));
    auto nine = context.mark_node(v0::Constant::create(element::f32, Shape{}, {9.f}));
    auto min_uniform =
        context.mark_node(v0::Constant::create(element::f32, Shape{}, {min_uniform_value}));

    auto lt_one_mask = context.mark_node(std::make_shared<v1::Less>(concentration, one));
    auto conc_plus_one = context.mark_node(std::make_shared<v1::Add>(concentration, one));
    auto conc_ge_one = context.mark_node(std::make_shared<v1::Select>(lt_one_mask, conc_plus_one, concentration));

    auto d = context.mark_node(std::make_shared<v1::Subtract>(conc_ge_one, third));
    auto nine_d = context.mark_node(std::make_shared<v1::Multiply>(d, nine));
    auto sqrt_term = context.mark_node(std::make_shared<v0::Sqrt>(nine_d));
    auto c = context.mark_node(std::make_shared<v1::Divide>(one, sqrt_term));

    auto scale = one;
    auto mean = zero;
    auto normals = make_random_normal(context, expanded_shape, element::f32, scale, mean)[0];
    auto uniform_accept =
        context.mark_node(std::make_shared<v8::RandomUniform>(expanded_shape, min_uniform, one, element::f32));

    auto zero_bc = context.mark_node(std::make_shared<v3::Broadcast>(zero, expanded_shape));
    auto one_bc = context.mark_node(std::make_shared<v3::Broadcast>(one, expanded_shape));
    auto min_uniform_bc =
        context.mark_node(std::make_shared<v3::Broadcast>(min_uniform, expanded_shape));
    auto d_bc = context.mark_node(std::make_shared<v3::Broadcast>(d, expanded_shape));
    auto c_bc = context.mark_node(std::make_shared<v3::Broadcast>(c, expanded_shape));

    auto cx = context.mark_node(std::make_shared<v1::Multiply>(c_bc, normals));
    auto one_plus_cx = context.mark_node(std::make_shared<v1::Add>(one_bc, cx));
    auto v = context.mark_node(std::make_shared<v1::Multiply>(one_plus_cx, one_plus_cx));
    v = context.mark_node(std::make_shared<v1::Multiply>(v, one_plus_cx));
    auto safe_v = context.mark_node(std::make_shared<v1::Maximum>(v, min_uniform_bc));

    auto log_v = context.mark_node(std::make_shared<v0::Log>(safe_v));
    auto log_u = context.mark_node(std::make_shared<v0::Log>(uniform_accept));
    auto x_sq = context.mark_node(std::make_shared<v1::Multiply>(normals, normals));
    auto x_sq_half = context.mark_node(std::make_shared<v1::Multiply>(x_sq, half));

    auto d_times_v = context.mark_node(std::make_shared<v1::Multiply>(d_bc, v));
    auto d_minus_dv = context.mark_node(std::make_shared<v1::Subtract>(d_bc, d_times_v));
    auto d_log_v = context.mark_node(std::make_shared<v1::Multiply>(d_bc, log_v));
    auto rhs = context.mark_node(std::make_shared<v1::Add>(x_sq_half, d_minus_dv));
    rhs = context.mark_node(std::make_shared<v1::Add>(rhs, d_log_v));

    auto positive_mask = context.mark_node(std::make_shared<v1::Greater>(v, zero_bc));
    auto compare_mask = context.mark_node(std::make_shared<v1::Less>(log_u, rhs));
    auto accept_mask = context.mark_node(std::make_shared<v1::LogicalAnd>(positive_mask, compare_mask));

    auto candidate = context.mark_node(std::make_shared<v1::Multiply>(d_bc, v));
    auto accept_i32 = context.mark_node(std::make_shared<v0::Convert>(accept_mask, element::i32));
    auto prefix = context.mark_node(std::make_shared<v0::CumSum>(accept_i32, axis_zero_i32, false, false));
    auto one_i32 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto one_i32_bc = context.mark_node(std::make_shared<v3::Broadcast>(one_i32, expanded_shape));
    auto first_accept = context.mark_node(
        std::make_shared<v1::LogicalAnd>(accept_mask,
                                         context.mark_node(std::make_shared<v1::Equal>(prefix, one_i32_bc))));

    auto first_accept_f = context.mark_node(std::make_shared<v0::Convert>(first_accept, element::f32));
    auto selected = context.mark_node(std::make_shared<v1::Multiply>(candidate, first_accept_f));
    auto gamma_candidates =
        context.mark_node(std::make_shared<v1::ReduceSum>(selected, axis_zero_i64, false));
    auto any_accept =
        context.mark_node(std::make_shared<v1::ReduceLogicalOr>(accept_mask, axis_zero_i64, false));

    auto last_index =
        context.mark_node(v0::Constant::create(element::i64, Shape{}, {standard_gamma_trials - 1}));
    auto last_candidate =
        context.mark_node(std::make_shared<v8::Gather>(candidate, last_index, axis_zero_i64));
    auto gamma_base =
        context.mark_node(std::make_shared<v1::Select>(any_accept, gamma_candidates, last_candidate));

    auto frac_uniform =
        context.mark_node(std::make_shared<v8::RandomUniform>(shape, min_uniform, one, element::f32));
    auto safe_alpha = context.mark_node(std::make_shared<v1::Maximum>(concentration, min_uniform));
    auto alpha_for_inv = context.mark_node(std::make_shared<v1::Select>(lt_one_mask, safe_alpha, one));
    auto inv_alpha = context.mark_node(std::make_shared<v1::Divide>(one, alpha_for_inv));
    auto pow_term = context.mark_node(std::make_shared<v1::Power>(frac_uniform, inv_alpha));
    auto adjusted = context.mark_node(std::make_shared<v1::Multiply>(gamma_base, pow_term));
    auto gamma_fp32 = context.mark_node(std::make_shared<v1::Select>(lt_one_mask, adjusted, gamma_base));

    Output<Node> result = gamma_fp32;
    if (output_type != element::f32) {
        result = context.mark_node(std::make_shared<v1::ConvertLike>(result, input));
    }
    return {result};
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
            if (ov::as_type_ptr<v0::Constant>(context.get_input_from_visible_context(3).get_node_shared_ptr())) {
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
