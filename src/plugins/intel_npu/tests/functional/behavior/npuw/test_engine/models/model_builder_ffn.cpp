// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_builder_ffn.hpp"

#include <vector>

#include "model_builder.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/opsets/opset11.hpp"

namespace ov {
namespace test {
namespace npuw {

ov::Output<ov::Node> SwiGLU::operator()(const ov::Output<ov::Node>& input, const std::string& name) const {
    auto gate = make_linear(input, hidden_size, intermediate_size, name + ".gate_proj", precision, weight_fn);
    auto up = make_linear(input, hidden_size, intermediate_size, name + ".up_proj", precision, weight_fn);

    auto sigmoid = std::make_shared<ov::opset11::Sigmoid>(gate);

    auto silu = std::make_shared<ov::opset11::Multiply>(gate, sigmoid);
    silu->set_friendly_name(name + "_silu");

    auto gate_up = std::make_shared<ov::opset11::Multiply>(silu, up);
    gate_up->set_friendly_name(name + "_gate_up");

    auto down = make_linear(gate_up, intermediate_size, hidden_size, name + ".down_proj", precision, weight_fn);

    return down;
}

ov::Output<ov::Node> GELU::operator()(const ov::Output<ov::Node>& input, const std::string& name) const {
    auto up = make_linear(input, hidden_size, intermediate_size, name + ".up_proj", precision, weight_fn, bias_fn);

    auto gelu = std::make_shared<ov::opset11::Gelu>(up);
    gelu->set_friendly_name(name + "_gelu");

    auto down = make_linear(gelu, intermediate_size, hidden_size, name + ".down_proj", precision, weight_fn, bias_fn);

    return down;
}

MoEFFN::MoEFFN(size_t hs, size_t is, size_t ne, size_t k, ov::element::Type prec, WeightFn wf)
    : hidden_size(hs),
      intermediate_size(is),
      num_experts(ne),
      num_experts_per_tok(k),
      precision(prec),
      weight_fn(std::move(wf)) {
    // Default to i4 CompressedWeight if no weight function provided.
    // i4 (not nf4) because NPUW's nf4 unpack doesn't handle 3D batched scales.
    if (!weight_fn) {
        weight_fn = CompressedWeight{ov::element::i4, 0, DCOffPattern::SYMM_NO_ZP};
    }
    using C = ov::opset11::Constant;
    int32_t is_i = static_cast<int32_t>(is);
    int32_t ne_i = static_cast<int32_t>(ne);
    int32_t k_i = static_cast<int32_t>(k);

    // Use i32 for shape/axis/step constants matching real GPT-OSS HuggingFace export.
    tile_repeats = C::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{ne_i, 1});
    slice_step = C::create(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{1});
    slice_axis2 = C::create(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{2});
    slice_start_0 = C::create(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{0});
    slice_stop_is = C::create(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{is_i});
    slice_start_is = C::create(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{is_i});
    slice_stop_2is = C::create(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{static_cast<int32_t>(2 * is_i)});
    min_const = C::create(prec, ov::Shape{1}, std::vector<float>{20.0f});
    swish_beta = C::create(prec, ov::Shape{}, std::vector<float>{1.0f});
    clamp_add_zero = C::create(prec, ov::Shape{1}, std::vector<float>{0.0f});
    topk_k_const = C::create(ov::element::i32, ov::Shape{}, std::vector<int32_t>{k_i});
    sl_start = C::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{0, 0});
    sl_step_r = C::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{1, 1});
    sl_axes = C::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{0, 1});
    scatter_axis = C::create(ov::element::i32, ov::Shape{}, std::vector<int32_t>{1});
    tp_order = C::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{1, 0});
    unsq_axis = C::create(ov::element::i32, ov::Shape{}, std::vector<int32_t>{3});
    reduce_axis = C::create(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{0});
}

ov::Output<ov::Node> MoEFFN::operator()(const ov::Output<ov::Node>& input, const std::string& name) const {
    using C = ov::opset11::Constant;
    const auto prec = precision;
    const int32_t ne_i = static_cast<int32_t>(num_experts);
    const int32_t hs_i = static_cast<int32_t>(hidden_size);
    auto mk = [](std::vector<int32_t> v) {
        ov::OutputVector p;
        for (auto x : v)
            p.push_back(C::create(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{x})->output(0));
        return std::make_shared<ov::opset11::Concat>(p, 0);
    };

    auto original_shape = std::make_shared<ov::opset11::ShapeOf>(input, ov::element::i32);
    original_shape->set_friendly_name(name + ".original_shape");

    auto input_2d = std::make_shared<ov::opset11::Reshape>(input, mk({-1, hs_i}), false);
    input_2d->set_friendly_name(name + ".input_2d");

    // Router uses i8 weights matching real GPT-OSS two-pass quantization
    // (router excluded from 4-bit, gets 8-bit instead).
    static const CompressedWeight router_wt{ov::element::i8, 0, DCOffPattern::SYMM_NO_ZP};
    auto rw = router_wt(name + ".expert.router.weight", ov::Shape{num_experts, hidden_size}, prec);
    auto r_mm = std::make_shared<ov::opset11::MatMul>(input_2d, rw, false, true);
    r_mm->set_friendly_name(name + ".expert.router.matmul");
    auto r_bias = C::create(prec, ov::Shape{1, num_experts}, std::vector<float>(num_experts, 0.0f));
    r_bias->set_friendly_name(name + ".expert.router.bias");
    auto r_add = std::make_shared<ov::opset11::Add>(r_mm, r_bias);
    r_add->set_friendly_name(name + ".expert.router.add");

    auto topk = std::make_shared<ov::opset11::TopK>(r_add, topk_k_const, 1, "max", "value", ov::element::i64);
    topk->set_friendly_name(name + ".expert.router.topk");
    auto softmax = std::make_shared<ov::op::v8::Softmax>(topk->output(0), 1);
    softmax->set_friendly_name(name + ".expert.router.softmax");
    auto topk_cvt = std::make_shared<ov::opset11::Convert>(topk->output(1), ov::element::i32);
    topk_cvt->set_friendly_name(name + ".expert.router.topk_convert");
    auto topk_shape = std::make_shared<ov::op::v3::ShapeOf>(topk_cvt, ov::element::i32);
    topk_shape->set_friendly_name(name + ".expert.router.topk_shapeof");

    auto r_slice = std::make_shared<ov::op::v8::Slice>(softmax, sl_start, topk_shape, sl_step_r, sl_axes);
    r_slice->set_friendly_name(name + ".expert.router.slice");
    auto add_shape = std::make_shared<ov::op::v3::ShapeOf>(r_add, ov::element::i32);
    add_shape->set_friendly_name(name + ".expert.router.add_shapeof");
    auto zeros = std::make_shared<ov::op::v3::Broadcast>(C::create(prec, ov::Shape{}, std::vector<float>{0.0f}),
                                                         add_shape,
                                                         ov::op::BroadcastType::NUMPY);
    zeros->set_friendly_name(name + ".expert.router.zeros");
    auto scatter = std::make_shared<ov::op::v3::ScatterElementsUpdate>(zeros, topk_cvt, r_slice, scatter_axis);
    scatter->set_friendly_name(name + ".expert.router.scatter");

    auto r_tp = std::make_shared<ov::opset11::Transpose>(scatter, tp_order);
    r_tp->set_friendly_name(name + ".expert.router.transpose");
    auto r_reshape = std::make_shared<ov::opset11::Reshape>(r_tp, mk({ne_i, 1, -1}), false);
    r_reshape->set_friendly_name(name + ".expert.router.reshape");
    auto router_scores = std::make_shared<ov::opset11::Unsqueeze>(r_reshape, unsq_axis);
    router_scores->set_friendly_name(name + ".expert.router.unsqueeze");

    // Expert: Tile → Reshape → MatMul → Add → dual-branch → MatMul → Add → Reshape → Multiply → ReduceSum
    auto tiled = std::make_shared<ov::op::v0::Tile>(input_2d, tile_repeats);
    tiled->set_friendly_name(name + ".expert.tile");
    auto expert_3d = std::make_shared<ov::opset11::Reshape>(tiled, mk({ne_i, -1, hs_i}), false);
    expert_3d->set_friendly_name(name + ".expert.reshape_in");

    auto gu_w = weight_fn(name + ".expert.gate_up_proj.weight",
                          ov::Shape{num_experts, 2 * intermediate_size, hidden_size},
                          prec);
    auto gu_mm = std::make_shared<ov::opset11::MatMul>(expert_3d, gu_w, false, true);
    gu_mm->set_friendly_name(name + ".expert.gate_up_matmul");
    auto gu_bias = C::create(prec,
                             ov::Shape{num_experts, 1, 2 * intermediate_size},
                             std::vector<float>(num_experts * 2 * intermediate_size, 0.0f));
    gu_bias->set_friendly_name(name + ".expert.gate_up_bias");
    auto gu_add = std::make_shared<ov::opset11::Add>(gu_mm, gu_bias);
    gu_add->set_friendly_name(name + ".expert.gate_up_add");

    auto act_slice = std::make_shared<ov::op::v8::Slice>(gu_add, slice_start_0, slice_stop_is, slice_step, slice_axis2);
    act_slice->set_friendly_name(name + ".expert.slice_act");
    auto act_min = std::make_shared<ov::opset11::Minimum>(act_slice, min_const);
    act_min->set_friendly_name(name + ".expert.minimum");
    auto act_swish = std::make_shared<ov::op::v4::Swish>(act_min, swish_beta);
    act_swish->set_friendly_name(name + ".expert.swish");

    auto gate_slice =
        std::make_shared<ov::op::v8::Slice>(gu_add, slice_start_is, slice_stop_2is, slice_step, slice_axis2);
    gate_slice->set_friendly_name(name + ".expert.slice_gate");
    auto gate_clamp = std::make_shared<ov::op::v0::Clamp>(gate_slice, -20.0f, 20.0f);
    gate_clamp->set_friendly_name(name + ".expert.clamp");
    auto gate_add = std::make_shared<ov::opset11::Add>(gate_clamp, clamp_add_zero);
    gate_add->set_friendly_name(name + ".expert.gate_add");

    auto merged = std::make_shared<ov::opset11::Multiply>(act_swish, gate_add);
    merged->set_friendly_name(name + ".expert.merge");

    auto dn_w =
        weight_fn(name + ".expert.down_proj.weight", ov::Shape{num_experts, hidden_size, intermediate_size}, prec);
    auto dn_mm = std::make_shared<ov::opset11::MatMul>(merged, dn_w, false, true);
    dn_mm->set_friendly_name(name + ".expert.down_matmul");
    auto dn_bias =
        C::create(prec, ov::Shape{num_experts, 1, hidden_size}, std::vector<float>(num_experts * hidden_size, 0.0f));
    dn_bias->set_friendly_name(name + ".expert.down_bias");
    auto dn_add = std::make_shared<ov::opset11::Add>(dn_mm, dn_bias);
    dn_add->set_friendly_name(name + ".expert.down_add");

    auto expert_out = std::make_shared<ov::opset11::Reshape>(dn_add, mk({ne_i, 1, -1, hs_i}), false);
    expert_out->set_friendly_name(name + ".expert.reshape_out");
    auto weighted = std::make_shared<ov::opset11::Multiply>(expert_out, router_scores);
    weighted->set_friendly_name(name + ".expert.weighted");
    auto reduced = std::make_shared<ov::opset11::ReduceSum>(weighted, reduce_axis, false);
    reduced->set_friendly_name(name + ".expert.reduced");

    auto output = std::make_shared<ov::opset11::Reshape>(reduced, original_shape, false);
    output->set_friendly_name(name + ".output");
    return output->output(0);
}

}  // namespace npuw
}  // namespace test
}  // namespace ov
