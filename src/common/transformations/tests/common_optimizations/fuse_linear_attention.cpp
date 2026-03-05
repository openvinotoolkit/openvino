// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_linear_attention.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/linear_attn.hpp"
#include "openvino/opsets/opset1.hpp"

using namespace testing;
using namespace ov;
using namespace ov::op;

namespace {

std::shared_ptr<ov::Node> build_l2_norm(const ov::Output<ov::Node>& input) {
    auto square = std::make_shared<v1::Multiply>(input, input, AutoBroadcastType::NUMPY);
    auto axis = v0::Constant::create(element::i32, Shape{1}, {3});
    auto reduce = std::make_shared<v1::ReduceSum>(square, axis, true);
    auto eps = v0::Constant::create(element::f32, Shape{1, 1, 1, 1}, {1e-6f});
    auto add = std::make_shared<v1::Add>(reduce, eps, AutoBroadcastType::NUMPY);
    auto sqrt = std::make_shared<v0::Sqrt>(add);
    auto one = v0::Constant::create(element::f32, Shape{1, 1, 1, 1}, {1.0f});
    auto inv = std::make_shared<v1::Divide>(one, sqrt, AutoBroadcastType::NUMPY);
    return std::make_shared<v1::Multiply>(input, inv, AutoBroadcastType::NUMPY);
}

std::shared_ptr<ov::Model> build_linear_attention_loop_model() {
    auto query = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 16, 128});
    query->set_friendly_name("query");
    auto key = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 16, 128});
    key->set_friendly_name("key");
    auto value = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 16, 128});
    value->set_friendly_name("value");
    auto g = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 16});
    g->set_friendly_name("g");
    auto beta = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 16});
    beta->set_friendly_name("beta");
    auto initial_state = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 16, 128, 128});
    initial_state->set_friendly_name("initial_state");
    auto init_output = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 16, 128});
    init_output->set_friendly_name("init_output");

    auto trip_count = v0::Constant::create(element::i32, Shape{}, {8});
    auto condition = v0::Constant::create(element::boolean, Shape{}, {true});
    auto loop = std::make_shared<v5::Loop>(trip_count, condition);

    auto body_out_accum = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 16, 128});
    auto body_state = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 16, 128, 128});
    auto body_beta = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 1, 16});
    auto body_g = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 1, 16});
    auto body_value = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 1, 16, 128});
    auto body_key = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 1, 16, 128});
    auto body_query = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 1, 16, 128});
    auto body_iter = std::make_shared<v0::Parameter>(element::i32, PartialShape{});

    auto axis0 = v0::Constant::create(element::i32, Shape{1}, {0});
    auto axis1 = v0::Constant::create(element::i32, Shape{1}, {1});
    auto axis2 = v0::Constant::create(element::i32, Shape{1}, {2});
    auto axis3 = v0::Constant::create(element::i32, Shape{1}, {3});

    auto beta_sq = std::make_shared<v0::Squeeze>(body_beta, axis1);
    auto beta_unsq0 = std::make_shared<v0::Unsqueeze>(beta_sq, axis2);
    auto beta_unsq1 = std::make_shared<v0::Unsqueeze>(beta_unsq0, axis3);
    auto beta_exp = std::make_shared<v0::Exp>(beta_unsq1);

    auto state_scaled = std::make_shared<v1::Multiply>(body_state, beta_exp, AutoBroadcastType::NUMPY);

    auto query_sq = std::make_shared<v0::Squeeze>(body_query, axis1);
    auto key_sq = std::make_shared<v0::Squeeze>(body_key, axis1);
    auto query_unsq = std::make_shared<v0::Unsqueeze>(query_sq, axis3);
    auto key_unsq = std::make_shared<v0::Unsqueeze>(key_sq, axis2);
    auto kv = std::make_shared<v1::Multiply>(query_unsq, key_unsq, AutoBroadcastType::NUMPY);

    auto state_sum = std::make_shared<v1::ReduceSum>(state_scaled, axis3, false);
    auto state_add = std::make_shared<v1::Add>(state_scaled, kv, AutoBroadcastType::NUMPY);

    auto value_sq = std::make_shared<v0::Squeeze>(body_value, axis1);
    auto g_sq = std::make_shared<v0::Squeeze>(body_g, axis1);
    auto g_unsq = std::make_shared<v0::Unsqueeze>(g_sq, axis2);
    auto scaled_value = std::make_shared<v1::Multiply>(value_sq, g_unsq, AutoBroadcastType::NUMPY);
    auto state_sum_unsq = std::make_shared<v0::Unsqueeze>(state_sum, axis1);
    auto updates_base = std::make_shared<v0::Unsqueeze>(scaled_value, axis1);
    auto updates = std::make_shared<v1::Add>(updates_base, state_sum_unsq, AutoBroadcastType::NUMPY);

    auto scatter_indices = v0::Constant::create(element::i32, Shape{1}, {0});
    auto scatter_axis = v0::Constant::create(element::i32, Shape{1}, {1});
    auto scatter = std::make_shared<v3::ScatterUpdate>(body_out_accum, scatter_indices, updates, scatter_axis);

    auto cond_res = std::make_shared<v0::Result>(condition);
    auto result_out = std::make_shared<v0::Result>(scatter);
    auto result_state = std::make_shared<v0::Result>(state_add);

    auto body = std::make_shared<Model>(OutputVector{cond_res, result_out, result_state},
                                        ParameterVector{body_out_accum,
                                                       body_state,
                                                       body_beta,
                                                       body_g,
                                                       body_value,
                                                       body_key,
                                                       body_query,
                                                       body_iter});
    loop->set_special_body_ports({7, 0});
    loop->set_function(body);

    auto query_norm = build_l2_norm(query);
    auto value_norm = build_l2_norm(value);

    loop->set_sliced_input(body_query, query_norm, 0, 1, 1, -1, 1);
    loop->set_sliced_input(body_key, key, 0, 1, 1, -1, 1);
    loop->set_sliced_input(body_value, value_norm, 0, 1, 1, -1, 1);
    loop->set_sliced_input(body_g, g, 0, 1, 1, -1, 1);
    loop->set_sliced_input(body_beta, beta, 0, 1, 1, -1, 1);
    loop->set_merged_input(body_state, initial_state, result_state);
    loop->set_merged_input(body_out_accum, init_output, result_out);

    auto loop_out0 = loop->get_concatenated_slices(result_out, 0, 1, 1, -1, 1);
    auto loop_out1 = loop->get_iter_value(result_state);

    auto out0 = std::make_shared<v0::Result>(loop_out0);
    auto out1 = std::make_shared<v0::Result>(loop_out1);

    return std::make_shared<Model>(OutputVector{out0, out1},
                                   ParameterVector{query, key, value, g, beta, initial_state, init_output});
}

std::shared_ptr<ov::Model> build_linear_attention_ref_model() {
    auto query = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 16, 128});
    query->set_friendly_name("query");
    auto key = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 16, 128});
    key->set_friendly_name("key");
    auto value = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 16, 128});
    value->set_friendly_name("value");
    auto g = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 16});
    g->set_friendly_name("g");
    auto beta = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 16});
    beta->set_friendly_name("beta");
    auto initial_state = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 16, 128, 128});
    initial_state->set_friendly_name("initial_state");
    auto init_output = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, 16, 128});
    init_output->set_friendly_name("init_output");

    auto linear_attn = std::make_shared<ov::op::LinearAttention>(
        OutputVector{query, key, value, g, beta, initial_state});

    auto out0 = std::make_shared<v0::Result>(linear_attn->output(0));
    auto out1 = std::make_shared<v0::Result>(linear_attn->output(1));

    return std::make_shared<Model>(OutputVector{out0, out1},
                                   ParameterVector{query, key, value, g, beta, initial_state, init_output});
}

}  // namespace

TEST_F(TransformationTestsF, FuseLinearAttention_LoopSubgraph) {
    disable_rt_info_check();
    model = build_linear_attention_loop_model();
    manager.register_pass<ov::pass::LinearAttentionFusion>();

    model_ref = build_linear_attention_ref_model();
}
