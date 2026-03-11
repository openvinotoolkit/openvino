// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/gated_delta_net_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/gated_delta_net.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v5 = ov::op::v5;
namespace v8 = ov::op::v8;

static constexpr int64_t B = 2;
static constexpr int64_t H = 4;
static constexpr int64_t S = 3;
static constexpr int64_t Dk = 8;
static constexpr int64_t Dv = 16;

static std::shared_ptr<Node> build_exporter_concat_output(const Output<Node>& query,
                                                          const Output<Node>& key,
                                                          const Output<Node>& value,
                                                          const Output<Node>& recurrent_state,
                                                          const Output<Node>& gate,
                                                          const Output<Node>& beta,
                                                          bool core_attn_keep_dims = true,
                                                          bool flatten_outputs = true) {
    const auto const_zero_i32 = v0::Constant::create(element::i32, Shape{}, {0});
    const auto const_two_i32 = v0::Constant::create(element::i32, Shape{}, {2});
    const auto const_minus_one = v0::Constant::create(element::i32, Shape{}, {-1});
    const auto const_minus_two = v0::Constant::create(element::i32, Shape{}, {-2});
    const auto flatten_shape = flatten_outputs ? v0::Constant::create(element::i32, Shape{1}, {-1})
                                               : v0::Constant::create(element::i32, Shape{2}, {-1, 1});

    const auto value_shape = std::make_shared<v3::ShapeOf>(value, element::i32);
    const auto zero_f32 = v0::Constant::create(element::f32, Shape{}, {0.0f});
    const auto core_attn_out = std::make_shared<v3::Broadcast>(zero_f32, value_shape);
    const auto seq_len = std::make_shared<v8::Gather>(value_shape, const_two_i32, const_zero_i32);
    const auto seq_len_i32 = std::make_shared<v0::Convert>(seq_len, element::i32);
    const auto exec_cond = v0::Constant::create(element::boolean, Shape{}, {true});

    auto timestep_param = std::make_shared<v0::Parameter>(element::i32, PartialShape{});
    auto q_t_param = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 1, -1});
    auto k_t_param = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 1, -1});
    auto v_t_param = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 1, -1});
    auto g_t_param = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 1});
    auto beta_t_param = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 1});
    auto last_state_param = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto core_attn_param = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});

    auto q_t = std::make_shared<v0::Squeeze>(q_t_param, const_two_i32);
    auto k_t = std::make_shared<v0::Squeeze>(k_t_param, const_two_i32);
    auto v_t = std::make_shared<v0::Squeeze>(v_t_param, const_two_i32);
    auto g_t = std::make_shared<v0::Unsqueeze>(std::make_shared<v0::Exp>(g_t_param), const_minus_one);

    auto last_state_in = std::make_shared<v1::Multiply>(last_state_param, g_t);
    std::shared_ptr<Node> kv_mem =
        std::make_shared<v1::Multiply>(last_state_in, std::make_shared<v0::Unsqueeze>(k_t, const_minus_one));
    kv_mem = std::make_shared<v1::ReduceSum>(kv_mem, const_minus_two, false);

    auto delta = std::make_shared<v1::Multiply>(std::make_shared<v1::Subtract>(v_t, kv_mem), beta_t_param);
    auto last_state_delta = std::make_shared<v1::Multiply>(std::make_shared<v0::Unsqueeze>(k_t, const_minus_one),
                                                           std::make_shared<v0::Unsqueeze>(delta, const_minus_two));
    auto last_state_next = std::make_shared<v1::Add>(last_state_in, last_state_delta);
    std::shared_ptr<Node> core_attn_update =
        std::make_shared<v1::Multiply>(last_state_next, std::make_shared<v0::Unsqueeze>(q_t, const_minus_one));
    core_attn_update = std::make_shared<v1::ReduceSum>(core_attn_update, const_minus_two, core_attn_keep_dims);

    auto timestep = std::make_shared<v0::Unsqueeze>(timestep_param, const_zero_i32);
    auto core_attn_res = std::make_shared<v3::ScatterUpdate>(core_attn_param, timestep, core_attn_update, const_two_i32);
    auto body_cond = v0::Constant::create(element::boolean, Shape{1}, {true});
    auto last_state_result = std::make_shared<v0::Result>(last_state_next);
    auto core_attn_result = std::make_shared<v0::Result>(core_attn_res);

    auto body = std::make_shared<Model>(OutputVector{body_cond, last_state_result, core_attn_result},
                                        ParameterVector{timestep_param,
                                                        q_t_param,
                                                        k_t_param,
                                                        v_t_param,
                                                        g_t_param,
                                                        beta_t_param,
                                                        last_state_param,
                                                        core_attn_param},
                                        "body_model");

    auto loop = std::make_shared<v5::Loop>(seq_len_i32, exec_cond);
    loop->set_function(body);
    loop->set_special_body_ports({0, 0});
    loop->set_sliced_input(q_t_param, query, 0, 1, 1, -1, 2);
    loop->set_sliced_input(k_t_param, key, 0, 1, 1, -1, 2);
    loop->set_sliced_input(v_t_param, value, 0, 1, 1, -1, 2);
    loop->set_sliced_input(g_t_param, gate, 0, 1, 1, -1, 2);
    loop->set_sliced_input(beta_t_param, beta, 0, 1, 1, -1, 2);
    loop->set_merged_input(last_state_param, recurrent_state, last_state_result);
    loop->set_merged_input(core_attn_param, core_attn_out, core_attn_result);

    auto core_attn_last = loop->get_iter_value(core_attn_result, -1);
    auto last_state_last = loop->get_iter_value(last_state_result, -1);
    auto core_attn_flat = std::make_shared<v1::Reshape>(core_attn_last, flatten_shape, false);
    auto last_state_flat = std::make_shared<v1::Reshape>(last_state_last, flatten_shape, false);
    return std::make_shared<v0::Concat>(OutputVector{core_attn_flat, last_state_flat}, 0);
}

static std::shared_ptr<Node> build_fused_concat_output(const Output<Node>& query,
                                                       const Output<Node>& key,
                                                       const Output<Node>& value,
                                                       const Output<Node>& recurrent_state,
                                                       const Output<Node>& gate,
                                                       const Output<Node>& beta) {
    const auto q_perm = v0::Constant::create(element::i64, Shape{4}, {0, 2, 1, 3});
    const auto g_perm = v0::Constant::create(element::i64, Shape{3}, {0, 2, 1});
    const auto flatten_shape = v0::Constant::create(element::i32, Shape{1}, {-1});

    auto q_seq = std::make_shared<v1::Transpose>(query, q_perm);
    auto k_seq = std::make_shared<v1::Transpose>(key, q_perm);
    auto v_seq = std::make_shared<v1::Transpose>(value, q_perm);
    auto g_seq = std::make_shared<v1::Transpose>(gate, g_perm);
    auto beta_seq = std::make_shared<v1::Transpose>(beta, g_perm);

    auto gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(q_seq, k_seq, v_seq, recurrent_state, g_seq, beta_seq);
    auto attn_head_first = std::make_shared<v1::Transpose>(gdn->output(0), q_perm);
    auto attn_flat = std::make_shared<v1::Reshape>(attn_head_first, flatten_shape, false);
    auto state_flat = std::make_shared<v1::Reshape>(gdn->output(1), flatten_shape, false);
    return std::make_shared<v0::Concat>(OutputVector{attn_flat, state_flat}, 0);
}

TEST_F(TransformationTestsF, GatedDeltaNetFusion_ExporterLoopStaticShapes) {
    {
        auto query = std::make_shared<v0::Parameter>(element::f32, Shape{B, H, S, Dk});
        auto key = std::make_shared<v0::Parameter>(element::f32, Shape{B, H, S, Dk});
        auto value = std::make_shared<v0::Parameter>(element::f32, Shape{B, H, S, Dv});
        auto recurrent_state = std::make_shared<v0::Parameter>(element::f32, Shape{B, H, Dk, Dv});
        auto gate = std::make_shared<v0::Parameter>(element::f32, Shape{B, H, S});
        auto beta = std::make_shared<v0::Parameter>(element::f32, Shape{B, H, S});

        auto concat_out = build_exporter_concat_output(query, key, value, recurrent_state, gate, beta);
        model = std::make_shared<Model>(OutputVector{concat_out},
                                        ParameterVector{query, key, value, recurrent_state, gate, beta});
        manager.register_pass<GatedDeltaNetFusion>();
    }
    {
        auto query = std::make_shared<v0::Parameter>(element::f32, Shape{B, H, S, Dk});
        auto key = std::make_shared<v0::Parameter>(element::f32, Shape{B, H, S, Dk});
        auto value = std::make_shared<v0::Parameter>(element::f32, Shape{B, H, S, Dv});
        auto recurrent_state = std::make_shared<v0::Parameter>(element::f32, Shape{B, H, Dk, Dv});
        auto gate = std::make_shared<v0::Parameter>(element::f32, Shape{B, H, S});
        auto beta = std::make_shared<v0::Parameter>(element::f32, Shape{B, H, S});

        auto concat_out = build_fused_concat_output(query, key, value, recurrent_state, gate, beta);
        model_ref = std::make_shared<Model>(OutputVector{concat_out},
                                            ParameterVector{query, key, value, recurrent_state, gate, beta});
    }
}

TEST_F(TransformationTestsF, GatedDeltaNetFusion_ExporterLoopDynamicShapes) {
    {
        auto query = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, H, -1, Dk});
        auto key = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, H, -1, Dk});
        auto value = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, H, -1, Dv});
        auto recurrent_state = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, H, Dk, Dv});
        auto gate = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, H, -1});
        auto beta = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, H, -1});

        auto concat_out = build_exporter_concat_output(query, key, value, recurrent_state, gate, beta);
        model = std::make_shared<Model>(OutputVector{concat_out},
                                        ParameterVector{query, key, value, recurrent_state, gate, beta});
        manager.register_pass<GatedDeltaNetFusion>();
    }
    {
        auto query = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, H, -1, Dk});
        auto key = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, H, -1, Dk});
        auto value = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, H, -1, Dv});
        auto recurrent_state = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, H, Dk, Dv});
        auto gate = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, H, -1});
        auto beta = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, H, -1});

        auto concat_out = build_fused_concat_output(query, key, value, recurrent_state, gate, beta);
        model_ref = std::make_shared<Model>(OutputVector{concat_out},
                                            ParameterVector{query, key, value, recurrent_state, gate, beta});
    }
}

TEST_F(TransformationTestsF, GatedDeltaNetFusion_NegativeNonFlatOutputReshape) {
    auto query = std::make_shared<v0::Parameter>(element::f32, Shape{B, H, S, Dk});
    auto key = std::make_shared<v0::Parameter>(element::f32, Shape{B, H, S, Dk});
    auto value = std::make_shared<v0::Parameter>(element::f32, Shape{B, H, S, Dv});
    auto recurrent_state = std::make_shared<v0::Parameter>(element::f32, Shape{B, H, Dk, Dv});
    auto gate = std::make_shared<v0::Parameter>(element::f32, Shape{B, H, S});
    auto beta = std::make_shared<v0::Parameter>(element::f32, Shape{B, H, S});

    auto concat_out = build_exporter_concat_output(query, key, value, recurrent_state, gate, beta, true, false);
    model = std::make_shared<Model>(OutputVector{concat_out},
                                    ParameterVector{query, key, value, recurrent_state, gate, beta});
    manager.register_pass<GatedDeltaNetFusion>();
}
