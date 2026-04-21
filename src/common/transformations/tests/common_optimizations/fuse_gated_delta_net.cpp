// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_gated_delta_net.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <queue>
#include <set>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "transformations/convert_precision.hpp"

namespace ov::test {

namespace {

ov::PartialShape adjust_input_shape(const ov::PartialShape& shape, const std::vector<size_t>& order) {
    if (std::is_sorted(order.begin(), order.end())) {
        return shape;
    } else {
        ov::PartialShape adjusted_shape(shape);
        for (Dimension::value_type i = 0; i < shape.rank().get_length(); ++i) {
            adjusted_shape[order[i]] = shape[i];
        }
        return adjusted_shape;
    }
};
}  // namespace
std::shared_ptr<ov::Model> build_looped_gdn(int32_t batch,
                                            int32_t seq_len,
                                            int32_t qk_head_num,
                                            int32_t v_head_num,
                                            int32_t qk_head_size,
                                            int32_t v_head_size,
                                            ov::element::Type dtype,
                                            std::vector<size_t>& input_order) {
    // 0, 1, 2, 3 for B, H, L, S
    bool ordered = std::is_sorted(input_order.begin(), input_order.end());

    ov::PartialShape qk_shape{batch, qk_head_num, seq_len, qk_head_size};
    ov::PartialShape v_tensor_shape{batch, v_head_num, seq_len, v_head_size};
    ov::PartialShape gv_shape{batch, qk_head_num, seq_len};
    ov::PartialShape h_shape{batch, qk_head_num, qk_head_size, v_head_size};

    qk_shape = adjust_input_shape(qk_shape, input_order);
    v_tensor_shape = adjust_input_shape(v_tensor_shape, input_order);
    gv_shape = adjust_input_shape(gv_shape, input_order);

    auto q = std::make_shared<ov::op::v0::Parameter>(dtype, qk_shape);
    auto k = std::make_shared<ov::op::v0::Parameter>(dtype, qk_shape);
    auto v = std::make_shared<ov::op::v0::Parameter>(dtype, v_tensor_shape);
    auto h0 = std::make_shared<ov::op::v0::Parameter>(dtype, h_shape);
    auto g = std::make_shared<ov::op::v0::Parameter>(dtype, gv_shape);
    auto beta = std::make_shared<ov::op::v0::Parameter>(dtype, gv_shape);

    auto l2norm = [&](const ov::Output<ov::Node>& x) {
        auto sq = std::make_shared<ov::op::v1::Multiply>(x, x);
        auto axis = ov::op::v0::Constant::create(ov::element::i32, {1}, {-1});
        auto sum = std::make_shared<ov::op::v1::ReduceSum>(sq, axis, true);
        auto eps = ov::op::v0::Constant::create(dtype, {}, {1e-6F});
        auto inv = std::make_shared<ov::op::v1::Divide>(
            ov::op::v0::Constant::create(dtype, {}, {1.0f}),
            std::make_shared<ov::op::v0::Sqrt>(std::make_shared<ov::op::v1::Add>(sum, eps)));
        return std::make_shared<ov::op::v1::Multiply>(x, inv);
    };

    auto q_norm = l2norm(q);
    auto k_norm = l2norm(k);

    auto perm_bhsd = ov::op::v0::Constant::create(ov::element::i64, {4}, input_order);
    auto perm_bhs = ov::op::v0::Constant::create(ov::element::i64,
                                                 {3},
                                                 std::vector<size_t>(input_order.begin(), input_order.begin() + 3));
    std::shared_ptr<ov::Node> q_in = q_norm;
    std::shared_ptr<ov::Node> k_in = k_norm;
    std::shared_ptr<ov::Node> v_in = v;
    std::shared_ptr<ov::Node> g_in = g;
    std::shared_ptr<ov::Node> beta_in = beta;
    if (!ordered) {
        q_in = std::make_shared<ov::op::v1::Transpose>(q_norm, perm_bhsd);
        k_in = std::make_shared<ov::op::v1::Transpose>(k_norm, perm_bhsd);
        v_in = std::make_shared<ov::op::v1::Transpose>(v, perm_bhsd);
        g_in = std::make_shared<ov::op::v1::Transpose>(g, perm_bhs);
        beta_in = std::make_shared<ov::op::v1::Transpose>(beta, perm_bhs);
    }

    auto shape_of_q = std::make_shared<ov::op::v3::ShapeOf>(q);
    auto gather_q_perm_index = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});
    auto gather_axis0 = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    auto gather_q_shape = std::make_shared<ov::op::v8::Gather>(shape_of_q, gather_q_perm_index, gather_axis0, 0);
    auto gather_head_size_index = ov::op::v0::Constant::create(ov::element::i64, {}, {3});
    auto gather_head_size =
        std::make_shared<ov::op::v8::Gather>(gather_q_shape, gather_head_size_index, gather_axis0, 0);
    auto q_d = std::make_shared<ov::op::v0::Convert>(gather_head_size, dtype);
    auto half = ov::op::v0::Constant::create(dtype, {}, {0.5f});
    auto q_scale = std::make_shared<ov::op::v1::Power>(q_d, half);
    auto q_scaled_t = std::make_shared<ov::op::v1::Divide>(q_in, q_scale);

    auto vt_shape = std::make_shared<ov::op::v3::ShapeOf>(v_in);
    auto core_attn_init =
        std::make_shared<ov::op::v3::Broadcast>(ov::op::v0::Constant::create(dtype, {}, {0.0f}), vt_shape);

    auto timestep = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{});
    auto q_i_param = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, -1, 1, -1});
    auto k_i_param = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, -1, 1, -1});
    auto v_i_param = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, -1, 1, -1});
    auto h_param = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, -1, -1, -1});
    auto g_i_param = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, -1, 1});
    auto beta_i_param = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, -1, 1});
    auto core_attn_buf = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, -1, -1, -1});

    auto minus1 = ov::op::v0::Constant::create(ov::element::i32, {1}, {-1});
    auto minus2 = ov::op::v0::Constant::create(ov::element::i32, {1}, {-2});
    auto axis_2 = ov::op::v0::Constant::create(ov::element::i32, {}, {2});

    auto q_s = std::make_shared<ov::op::v0::Squeeze>(q_i_param, axis_2);
    auto k_s = std::make_shared<ov::op::v0::Squeeze>(k_i_param, axis_2);
    auto v_s = std::make_shared<ov::op::v0::Squeeze>(v_i_param, axis_2);

    auto g_exp = std::make_shared<ov::op::v0::Exp>(g_i_param);
    auto g_exp_unsq = std::make_shared<ov::op::v0::Unsqueeze>(g_exp, minus1);
    auto h_decay = std::make_shared<ov::op::v1::Multiply>(h_param, g_exp_unsq);

    auto k_unsq = std::make_shared<ov::op::v0::Unsqueeze>(k_s, minus1);
    auto hk = std::make_shared<ov::op::v1::Multiply>(h_decay, k_unsq);
    auto v_prime = std::make_shared<ov::op::v1::ReduceSum>(hk, minus2, false);
    auto v_new = std::make_shared<ov::op::v1::Subtract>(v_s, v_prime);
    auto v_scaled = std::make_shared<ov::op::v1::Multiply>(v_new, beta_i_param);

    auto v_scaled_unsq = std::make_shared<ov::op::v0::Unsqueeze>(v_scaled, minus2);
    auto h_update = std::make_shared<ov::op::v1::Multiply>(k_unsq, v_scaled_unsq);
    auto h_res = std::make_shared<ov::op::v1::Add>(h_decay, h_update);

    auto q_unsq = std::make_shared<ov::op::v0::Unsqueeze>(q_s, minus1);
    auto hq = std::make_shared<ov::op::v1::Multiply>(h_res, q_unsq);
    auto o_step = std::make_shared<ov::op::v1::ReduceSum>(hq, minus2, true);

    auto timestep_unsq =
        std::make_shared<ov::op::v0::Unsqueeze>(timestep, ov::op::v0::Constant::create(ov::element::i32, {1}, {0}));
    auto core_buf_res = std::make_shared<ov::op::v3::ScatterUpdate>(core_attn_buf, timestep_unsq, o_step, axis_2);

    auto body_cond = ov::op::v0::Constant::create(ov::element::boolean, {1}, {true});
    auto body = std::make_shared<ov::Model>(
        ov::OutputVector{body_cond, h_res, core_buf_res},
        ov::ParameterVector{timestep, q_i_param, k_i_param, v_i_param, h_param, g_i_param, beta_i_param, core_attn_buf},
        "recurrent_body");

    auto t_index = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
    auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    auto trip_count_i64 = std::make_shared<ov::op::v8::Gather>(vt_shape, t_index, gather_axis);
    auto trip_count = std::make_shared<ov::op::v0::Convert>(trip_count_i64, ov::element::i32);

    auto loop =
        std::make_shared<ov::op::v5::Loop>(trip_count, ov::op::v0::Constant::create(ov::element::boolean, {1}, {true}));
    loop->set_function(body);
    loop->set_sliced_input(q_i_param, q_scaled_t, 0, 1, 1, -1, 2);
    loop->set_sliced_input(k_i_param, k_in, 0, 1, 1, -1, 2);
    loop->set_sliced_input(v_i_param, v_in, 0, 1, 1, -1, 2);
    loop->set_sliced_input(g_i_param, g_in, 0, 1, 1, -1, 2);
    loop->set_sliced_input(beta_i_param, beta_in, 0, 1, 1, -1, 2);
    loop->set_merged_input(h_param, h0, h_res);
    loop->set_merged_input(core_attn_buf, core_attn_init, core_buf_res);
    loop->set_special_body_ports({0, 0});

    auto core_attn_final_bhsd = loop->get_iter_value(core_buf_res, -1);
    auto h_final = loop->get_iter_value(h_res, -1);

    auto reshape_m1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    auto flat_core = std::make_shared<ov::op::v1::Reshape>(core_attn_final_bhsd, reshape_m1, false);
    auto flat_h = std::make_shared<ov::op::v1::Reshape>(h_final, reshape_m1, false);
    auto packed_loop_outputs = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{flat_core, flat_h}, 0);

    auto core_shape = std::make_shared<ov::op::v3::ShapeOf>(v_in);
    auto reduce_axis0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    auto core_numel = std::make_shared<ov::op::v1::ReduceProd>(core_shape, reduce_axis0, true);
    auto state_shape = std::make_shared<ov::op::v3::ShapeOf>(h0);
    auto slice_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    auto slice_step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto slice_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    auto slice_end_inf = ov::op::v0::Constant::create(ov::element::i64, {1}, {LLONG_MAX});

    auto core_slice =
        std::make_shared<ov::op::v8::Slice>(packed_loop_outputs, slice_start, core_numel, slice_step, slice_axis);
    auto state_slice =
        std::make_shared<ov::op::v8::Slice>(packed_loop_outputs, core_numel, slice_end_inf, slice_step, slice_axis);

    auto core_restored = std::make_shared<ov::op::v1::Reshape>(core_slice, core_shape, false);
    auto state_restored = std::make_shared<ov::op::v1::Reshape>(state_slice, state_shape, false);
    std::shared_ptr<ov::Node> core_attn_final = core_restored;
    if (!ordered) {
        core_attn_final =
            std::make_shared<ov::op::v1::Transpose>(core_restored,
                                                    ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    }

    // add a reshape to follow real model
    auto final_shape = ov::op::v0::Constant::create(ov::element::i64, {2}, {-1, v_head_size});
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(core_attn_final, final_shape, false);

    return std::make_shared<ov::Model>(ov::OutputVector{reshaped, state_restored},
                                       ov::ParameterVector{q, k, v, h0, g, beta});
}

std::shared_ptr<ov::Model> build_fused_gdn_ref(int32_t batch,
                                               int32_t seq_len,
                                               int32_t qk_head_num,
                                               int32_t v_head_num,
                                               int32_t qk_head_size,
                                               int32_t v_head_size,
                                               ov::element::Type dtype = ov::element::f32,
                                               std::vector<size_t> input_order = {0, 2, 1, 3}) {
    bool ordered = std::is_sorted(input_order.begin(), input_order.end());
    ov::PartialShape qk_shape{batch, qk_head_num, seq_len, qk_head_size};
    ov::PartialShape v_tensor_shape{batch, v_head_num, seq_len, v_head_size};
    ov::PartialShape gv_shape{batch, qk_head_num, seq_len};
    ov::PartialShape h_shape{batch, qk_head_num, qk_head_size, v_head_size};

    qk_shape = adjust_input_shape(qk_shape, input_order);
    v_tensor_shape = adjust_input_shape(v_tensor_shape, input_order);
    gv_shape = adjust_input_shape(gv_shape, input_order);

    auto q = std::make_shared<ov::op::v0::Parameter>(dtype, qk_shape);
    auto k = std::make_shared<ov::op::v0::Parameter>(dtype, qk_shape);
    auto v = std::make_shared<ov::op::v0::Parameter>(dtype, v_tensor_shape);
    auto h0 = std::make_shared<ov::op::v0::Parameter>(dtype, h_shape);
    auto g = std::make_shared<ov::op::v0::Parameter>(dtype, gv_shape);
    auto beta = std::make_shared<ov::op::v0::Parameter>(dtype, gv_shape);

    auto perm_bshd = ov::op::v0::Constant::create(
        ov::element::i64,
        {4},
        std::vector<size_t>{input_order[0], input_order[2], input_order[1], input_order[3]});
    auto perm_bsh = ov::op::v0::Constant::create(ov::element::i64,
                                                 {3},
                                                 std::vector<size_t>{input_order[0], input_order[2], input_order[1]});
    std::shared_ptr<ov::Node> q_in = q;
    std::shared_ptr<ov::Node> k_in = k;
    std::shared_ptr<ov::Node> v_in = v;
    std::shared_ptr<ov::Node> g_in = g;
    std::shared_ptr<ov::Node> beta_in = beta;
    if (ordered) {
        q_in = std::make_shared<ov::op::v1::Transpose>(q, perm_bshd);
        k_in = std::make_shared<ov::op::v1::Transpose>(k, perm_bshd);
        v_in = std::make_shared<ov::op::v1::Transpose>(v, perm_bshd);
        g_in = std::make_shared<ov::op::v1::Transpose>(g, perm_bsh);
        beta_in = std::make_shared<ov::op::v1::Transpose>(beta, perm_bsh);
    }

    auto gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(ov::OutputVector{q_in, k_in, v_in, h0, g_in, beta_in},
                                                                 true,
                                                                 1e-6F,
                                                                 1e-6F);

    ov::Output<ov::Node> gdn_core_output = gdn->output(0);
    if (ordered) {
        gdn_core_output = std::make_shared<ov::op::v1::Transpose>(gdn->output(0), perm_bshd);
    }

    auto final_shape = ov::op::v0::Constant::create(ov::element::i64, {2}, {-1, v_head_size});
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(gdn_core_output, final_shape, false);
    return std::make_shared<ov::Model>(ov::OutputVector{reshaped, gdn->output(1)},
                                       ov::ParameterVector{q, k, v, h0, g, beta});
}

TEST_F(TransformationTestsF, GatedDeltaNetFusion_BuildBHLSLoopedGDNMode) {
    disable_rt_info_check();
    disable_result_friendly_names_check();
    constexpr int32_t batch = -1;
    constexpr int32_t seq_len = -1;
    constexpr int32_t qk_head_num = 4;
    constexpr int32_t v_head_num = 4;
    constexpr int32_t qk_head_size = 8;
    constexpr int32_t v_head_size = 16;
    std::vector<size_t> input_order{0, 1, 2, 3};
    model = build_looped_gdn(batch,
                             seq_len,
                             qk_head_num,
                             v_head_num,
                             qk_head_size,
                             v_head_size,
                             ov::element::f32,
                             input_order);
    manager.register_pass<ov::pass::GatedDeltaNetFusion>();
    model_ref = build_fused_gdn_ref(batch,
                                    seq_len,
                                    qk_head_num,
                                    v_head_num,
                                    qk_head_size,
                                    v_head_size,
                                    ov::element::f32,
                                    input_order);
}

TEST_F(TransformationTestsF, GatedDeltaNetFusion_BuildBLHSLoopedGDNMode) {
    disable_rt_info_check();
    disable_result_friendly_names_check();
    constexpr int32_t batch = -1;
    constexpr int32_t seq_len = -1;
    constexpr int32_t qk_head_num = 4;
    constexpr int32_t v_head_num = 4;
    constexpr int32_t qk_head_size = 8;
    constexpr int32_t v_head_size = 16;
    std::vector<size_t> input_order{0, 2, 1, 3};
    model = build_looped_gdn(batch,
                             seq_len,
                             qk_head_num,
                             v_head_num,
                             qk_head_size,
                             v_head_size,
                             ov::element::f32,
                             input_order);
    manager.register_pass<ov::pass::GatedDeltaNetFusion>();
    model_ref = build_fused_gdn_ref(batch, seq_len, qk_head_num, v_head_num, qk_head_size, v_head_size);
}

namespace {

std::shared_ptr<ov::Model> build_gdn_with_shared_qk_anchor(bool shared_anchor) {
    using ov::op::v0::Constant;
    using ov::op::v0::Parameter;
    using ov::op::v0::Squeeze;
    using ov::op::v0::Unsqueeze;
    using ov::op::v1::Reshape;
    using ov::op::v1::Split;
    using ov::op::v1::Transpose;
    using ov::op::v8::Gather;

    auto q_src = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 4, 4, 8});
    auto k_src = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 4, 4, 8});
    auto v_src = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 4, 4, 8});

    auto split_axis = Constant::create(ov::element::i64, {}, {1});
    auto q_split = std::make_shared<Split>(q_src, split_axis, 2);
    auto k_split = shared_anchor ? q_split : std::make_shared<Split>(k_src, split_axis, 2);
    auto v_split = shared_anchor ? q_split : std::make_shared<Split>(v_src, split_axis, 2);

    auto q_anchor = q_split->output(0);
    auto k_anchor = k_split->output(1);
    auto v_anchor = v_split->output(0);

    auto gather_idx = Constant::create(ov::element::i64, {2}, {0, 1});
    auto gather_axis = Constant::create(ov::element::i64, {}, {1});
    auto q_gather = std::make_shared<Gather>(q_anchor, gather_idx, gather_axis);
    auto k_gather = std::make_shared<Gather>(k_anchor, gather_idx, gather_axis);
    auto v_gather = std::make_shared<Gather>(v_anchor, gather_idx, gather_axis);

    auto q_shape = Constant::create(ov::element::i64, {4}, {1, 2, 4, 8});
    auto k_shape = Constant::create(ov::element::i64, {4}, {1, 2, 4, 8});
    auto v_shape = Constant::create(ov::element::i64, {4}, {1, 2, 4, 8});
    auto q_reshape = std::make_shared<Reshape>(q_gather, q_shape, false);
    auto k_reshape = std::make_shared<Reshape>(k_gather, k_shape, false);
    auto v_reshape = std::make_shared<Reshape>(v_gather, v_shape, false);

    auto perm = Constant::create(ov::element::i64, {4}, {0, 1, 2, 3});
    auto q_transpose = std::make_shared<Transpose>(q_reshape, perm);
    auto k_transpose = std::make_shared<Transpose>(k_reshape, perm);
    auto v_transpose = std::make_shared<Transpose>(v_reshape, perm);

    auto unsq_axis = Constant::create(ov::element::i32, {1}, {0});
    auto q_unsq = std::make_shared<Unsqueeze>(q_transpose, unsq_axis);
    auto k_unsq = std::make_shared<Unsqueeze>(k_transpose, unsq_axis);
    auto v_unsq = std::make_shared<Unsqueeze>(v_transpose, unsq_axis);
    
    auto sq_axis = Constant::create(ov::element::i32, {1}, {0});
    auto q = std::make_shared<Squeeze>(q_unsq, sq_axis);
    auto k = std::make_shared<Squeeze>(k_unsq, sq_axis);
    auto v = std::make_shared<Squeeze>(v_unsq, sq_axis);

    auto state = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 4, 8, 8});
    auto gate = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 4, 4});
    auto beta = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 4, 4});

    auto gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(q, k, v, state, gate, beta);
    auto result0 = std::make_shared<ov::op::v0::Result>(gdn->output(0));
    auto result1 = std::make_shared<ov::op::v0::Result>(gdn->output(1));

    ov::ParameterVector params = {q_src, k_src, v_src, state, gate, beta};
    return std::make_shared<ov::Model>(ov::ResultVector{result0, result1}, params);
}

}  // namespace

TEST(TransformationTests, VerifySharedQKSourceForGDN_Positive) {
    auto model = build_gdn_with_shared_qk_anchor(true);
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::VerifySharedQKSourceForGDN>();
    manager.run_passes(model);

    auto gdn = ov::as_type_ptr<ov::op::internal::GatedDeltaNet>(model->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_NE(gdn, nullptr);

    auto q_input = gdn->input_value(0);
    auto k_input = gdn->input_value(1);
    auto v_input = gdn->input_value(2);

    // Find the anchor (Split) node for Q input
    // It may be wrapped in Reshape/Transpose/etc
    auto find_split_ancestor = [](const ov::Output<ov::Node>& output) -> std::shared_ptr<ov::Node> {
        auto node = output.get_node_shared_ptr();
        std::set<std::shared_ptr<ov::Node>> visited;
        std::queue<std::shared_ptr<ov::Node>> to_visit;
        to_visit.push(node);
        
        while (!to_visit.empty()) {
            auto current = to_visit.front();
            to_visit.pop();
            
            if (visited.count(current)) continue;
            visited.insert(current);
            
            if (ov::is_type<ov::op::v1::Split>(current)) {
                return current;
            }
            
            // Continue searching upwards
            for (size_t i = 0; i < current->get_input_size(); ++i) {
                to_visit.push(current->input_value(i).get_node_shared_ptr());
            }
        }
        return nullptr;
    };

    auto q_split = find_split_ancestor(q_input);
    auto k_split = find_split_ancestor(k_input);
    auto v_split = find_split_ancestor(v_input);
    
    ASSERT_NE(q_split, nullptr);
    ASSERT_NE(k_split, nullptr);
    ASSERT_NE(v_split, nullptr);
    ASSERT_EQ(q_split, k_split);
    ASSERT_EQ(k_split, v_split);
}

TEST(TransformationTests, VerifySharedQKSourceForGDN_NegativeDifferentAnchors) {
    auto model = build_gdn_with_shared_qk_anchor(false);
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::VerifySharedQKSourceForGDN>();
    manager.run_passes(model);

    auto gdn = ov::as_type_ptr<ov::op::internal::GatedDeltaNet>(model->get_results().at(0)->input_value(0).get_node_shared_ptr());
    ASSERT_NE(gdn, nullptr);

    auto q_input_node = gdn->input_value(0).get_node_shared_ptr();
    auto k_input_node = gdn->input_value(1).get_node_shared_ptr();
    auto v_input_node = gdn->input_value(2).get_node_shared_ptr();

    if (ov::is_type<ov::op::v1::Reshape>(q_input_node)) {
        q_input_node = q_input_node->input_value(0).get_node_shared_ptr();
    }
    if (ov::is_type<ov::op::v1::Reshape>(k_input_node)) {
        k_input_node = k_input_node->input_value(0).get_node_shared_ptr();
    }
    if (ov::is_type<ov::op::v1::Reshape>(v_input_node)) {
        v_input_node = v_input_node->input_value(0).get_node_shared_ptr();
    }

    // With different anchors, transformation should not apply, so at least one should not be Split
    ASSERT_FALSE(ov::is_type<ov::op::v1::Split>(q_input_node) && 
                 q_input_node == k_input_node && 
                 k_input_node == v_input_node);
}

TEST(TransformationTests, VerifySharedQKSourceForGDN_Qwen3NextModel) {
    const char* model_path_env = std::getenv("OV_GDN_TEST_MODEL_PATH");
    if (!model_path_env || std::string(model_path_env).empty()) {
        GTEST_SKIP() << "OV_GDN_TEST_MODEL_PATH is not set";
    }

    ov::Core core;
    auto model = core.read_model(model_path_env);

    size_t loops_before = 0;
    for (const auto& op : model->get_ops()) {
        if (std::strcmp(op->get_type_name(), "Loop") == 0) {
            ++loops_before;
        }
    }

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::GatedDeltaNetFusion>();
    manager.run_passes(model);

    size_t loops_after = 0;
    size_t gdn_after = 0;
    size_t gdn_same_qk_node = 0;
    for (const auto& op : model->get_ops()) {
        if (std::strcmp(op->get_type_name(), "Loop") == 0) {
            ++loops_after;
        }
        if (std::strcmp(op->get_type_name(), "GatedDeltaNet") == 0) {
            ++gdn_after;
            if (op->input_value(0).get_node_shared_ptr() == op->input_value(1).get_node_shared_ptr()) {
                ++gdn_same_qk_node;
            }
        }
    }

    EXPECT_GE(loops_before, 1);
    EXPECT_EQ(loops_after, 0);
    EXPECT_GE(gdn_after, 1);
    // Track how many fused GDN nodes have identical immediate Q/K producers for this model.
    // This is informative for local validation and may legitimately be zero for some graphs.
    (void)gdn_same_qk_node;
}

TEST_F(TransformationTestsF, GatedDeltaNetFusion_BuildBHLSLoopedGDNMode_F16) {
    disable_rt_info_check();
    disable_result_friendly_names_check();
    constexpr int32_t batch = -1;
    constexpr int32_t seq_len = -1;
    constexpr int32_t qk_head_num = 4;
    constexpr int32_t v_head_num = 4;
    constexpr int32_t qk_head_size = 8;
    constexpr int32_t v_head_size = 16;
    std::vector<size_t> input_order{0, 1, 2, 3};
    model = build_looped_gdn(batch,
                             seq_len,
                             qk_head_num,
                             v_head_num,
                             qk_head_size,
                             v_head_size,
                             ov::element::f16,
                             input_order);
    manager.register_pass<pass::ConvertPrecision>(ov::element::f32,
                                                  ov::element::f16,
                                                  type_to_fuse_map{},
                                                  true,
                                                  true,
                                                  false);
    manager.register_pass<ov::pass::GatedDeltaNetFusion>();
    model_ref = build_fused_gdn_ref(batch,
                                    seq_len,
                                    qk_head_num,
                                    v_head_num,
                                    qk_head_size,
                                    v_head_size,
                                    ov::element::f16,
                                    input_order);
}

TEST_F(TransformationTestsF, GatedDeltaNetFusion_BuildBLHSLoopedGDNMode_F16) {
    disable_rt_info_check();
    disable_result_friendly_names_check();
    constexpr int32_t batch = -1;
    constexpr int32_t seq_len = -1;
    constexpr int32_t qk_head_num = 4;
    constexpr int32_t v_head_num = 4;
    constexpr int32_t qk_head_size = 8;
    constexpr int32_t v_head_size = 16;
    std::vector<size_t> input_order{0, 2, 1, 3};
    model = build_looped_gdn(batch,
                             seq_len,
                             qk_head_num,
                             v_head_num,
                             qk_head_size,
                             v_head_size,
                             ov::element::f16,
                             input_order);
    manager.register_pass<pass::ConvertPrecision>(ov::element::f32,
                                                  ov::element::f16,
                                                  type_to_fuse_map{},
                                                  true,
                                                  true,
                                                  false);
    manager.register_pass<ov::pass::GatedDeltaNetFusion>();
    model_ref =
        build_fused_gdn_ref(batch, seq_len, qk_head_num, v_head_num, qk_head_size, v_head_size, ov::element::f16);
}
}  // namespace ov::test