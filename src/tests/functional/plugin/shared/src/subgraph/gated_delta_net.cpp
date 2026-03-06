// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/gated_delta_net.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"

#include <cmath>
#include <iostream>

namespace ov {
namespace test {

std::shared_ptr<ov::Model> GatedDeltaNet::buildLoopedGDN(int32_t batch,
                                                         int32_t seq_len,
                                                         int32_t qk_head_num,
                                                         int32_t v_head_num,
                                                         int32_t head_size) {
    const auto dtype = ov::element::f32;
    const ov::Shape qk_shape{static_cast<size_t>(batch),
                             static_cast<size_t>(seq_len),
                             static_cast<size_t>(qk_head_num),
                             static_cast<size_t>(head_size)};
    const ov::Shape v_tensor_shape{static_cast<size_t>(batch),
                                   static_cast<size_t>(seq_len),
                                   static_cast<size_t>(v_head_num),
                                   static_cast<size_t>(head_size)};
    const ov::Shape gv_shape{static_cast<size_t>(batch),
                             static_cast<size_t>(seq_len),
                             static_cast<size_t>(qk_head_num)};
    const ov::Shape h_shape{static_cast<size_t>(batch),
                            static_cast<size_t>(qk_head_num),
                            static_cast<size_t>(head_size),
                            static_cast<size_t>(head_size)};

    auto q = std::make_shared<ov::op::v0::Parameter>(dtype, qk_shape);
    auto k = std::make_shared<ov::op::v0::Parameter>(dtype, qk_shape);
    auto v = std::make_shared<ov::op::v0::Parameter>(dtype, v_tensor_shape);
    auto h0 = std::make_shared<ov::op::v0::Parameter>(dtype, h_shape);
    auto g = std::make_shared<ov::op::v0::Parameter>(dtype, gv_shape);
    auto beta = std::make_shared<ov::op::v0::Parameter>(dtype, gv_shape);

    q->set_friendly_name("q");
    k->set_friendly_name("k");
    v->set_friendly_name("v");
    h0->set_friendly_name("h0");
    g->set_friendly_name("g");
    beta->set_friendly_name("beta");

    auto l2norm = [&](const ov::Output<ov::Node>& x) {
        auto sq = std::make_shared<ov::op::v1::Multiply>(x, x);
        auto axis = ov::op::v0::Constant::create(ov::element::i32, {1}, {3});
        auto sum = std::make_shared<ov::op::v1::ReduceSum>(sq, axis, true);
        auto eps = ov::op::v0::Constant::create(dtype, {}, {1e-6f});
        auto inv = std::make_shared<ov::op::v1::Divide>(
            ov::op::v0::Constant::create(dtype, {}, {1.0f}),
            std::make_shared<ov::op::v0::Sqrt>(std::make_shared<ov::op::v1::Add>(sum, eps)));
        return std::make_shared<ov::op::v1::Multiply>(x, inv);
    };

    auto q_norm = l2norm(q);
    auto k_norm = l2norm(k);
    auto q_scale = ov::op::v0::Constant::create(dtype, {}, {1.0f / std::sqrt(static_cast<float>(head_size))});
    auto q_scaled = std::make_shared<ov::op::v1::Multiply>(q_norm, q_scale);

    auto v_shape = std::make_shared<ov::op::v3::ShapeOf>(v);
    auto core_attn_init =
        std::make_shared<ov::op::v3::Broadcast>(ov::op::v0::Constant::create(dtype, {}, {0.0f}), v_shape);

    auto timestep = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{});
    auto q_i_param = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, 1, -1, -1});
    auto k_i_param = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, 1, -1, -1});
    auto v_i_param = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, 1, -1, -1});
    auto h_param = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, -1, -1, -1});
    auto g_i_param = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, 1, -1});
    auto beta_i_param = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, 1, -1});
    auto core_attn_buf = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, -1, -1, -1});

    auto axis_time = ov::op::v0::Constant::create(ov::element::i32, {}, {1});
    auto b_q = std::make_shared<ov::op::v0::Squeeze>(q_i_param, axis_time);
    auto b_k = std::make_shared<ov::op::v0::Squeeze>(k_i_param, axis_time);
    auto b_v = std::make_shared<ov::op::v0::Squeeze>(v_i_param, axis_time);
    auto b_beta = std::make_shared<ov::op::v0::Squeeze>(beta_i_param, axis_time);
    auto b_g = std::make_shared<ov::op::v0::Squeeze>(g_i_param, axis_time);

    auto minus1 = ov::op::v0::Constant::create(ov::element::i32, {1}, {-1});
    auto g_unsq1 = std::make_shared<ov::op::v0::Unsqueeze>(b_g, minus1);
    auto g_unsq2 = std::make_shared<ov::op::v0::Unsqueeze>(g_unsq1, minus1);
    auto h_decay = std::make_shared<ov::op::v1::Multiply>(h_param, std::make_shared<ov::op::v0::Exp>(g_unsq2));

    auto b_k_unsq_v = std::make_shared<ov::op::v0::Unsqueeze>(b_k, minus1);
    auto v_prime = std::make_shared<ov::op::v1::ReduceSum>(
        std::make_shared<ov::op::v1::Multiply>(h_decay, b_k_unsq_v),
        ov::op::v0::Constant::create(ov::element::i32, {1}, {2}),
        false);
    auto v_new = std::make_shared<ov::op::v1::Subtract>(b_v, v_prime);
    auto v_scaled = std::make_shared<ov::op::v1::Multiply>(v_new, std::make_shared<ov::op::v0::Unsqueeze>(b_beta, minus1));

    auto v_scaled_unsq_k = std::make_shared<ov::op::v0::Unsqueeze>(v_scaled,
                                                                    ov::op::v0::Constant::create(ov::element::i32, {1}, {2}));
    auto h_update = std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v0::Unsqueeze>(b_k, minus1),
                                                           v_scaled_unsq_k);
    auto h_res = std::make_shared<ov::op::v1::Add>(h_decay, h_update);

    auto b_q_unsq_v = std::make_shared<ov::op::v0::Unsqueeze>(b_q, minus1);
    auto o_step = std::make_shared<ov::op::v1::ReduceSum>(
        std::make_shared<ov::op::v1::Multiply>(h_res, b_q_unsq_v),
        ov::op::v0::Constant::create(ov::element::i32, {1}, {2}),
        false);

    auto timestep_unsq = std::make_shared<ov::op::v0::Unsqueeze>(timestep,
                                                                 ov::op::v0::Constant::create(ov::element::i32, {1}, {0}));
    auto o_unsq = std::make_shared<ov::op::v0::Unsqueeze>(o_step, axis_time);
    auto core_buf_res = std::make_shared<ov::op::v3::ScatterUpdate>(core_attn_buf, timestep_unsq, o_unsq, axis_time);

    auto body_cond = ov::op::v0::Constant::create(ov::element::boolean, {1}, {true});
    auto body = std::make_shared<ov::Model>(
        ov::OutputVector{body_cond, h_res, core_buf_res},
        ov::ParameterVector{timestep, q_i_param, k_i_param, v_i_param, h_param, g_i_param, beta_i_param, core_attn_buf},
        "recurrent_body");

    auto t_index = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    auto trip_count_i64 = std::make_shared<ov::op::v8::Gather>(v_shape, t_index, gather_axis);
    auto trip_count = std::make_shared<ov::op::v0::Convert>(trip_count_i64, ov::element::i32);

    auto loop = std::make_shared<ov::op::v5::Loop>(trip_count,
                                                   ov::op::v0::Constant::create(ov::element::boolean, {1}, {true}));
    loop->set_function(body);
    loop->set_sliced_input(q_i_param, q_scaled, 0, 1, 1, -1, 1);
    loop->set_sliced_input(k_i_param, k_norm, 0, 1, 1, -1, 1);
    loop->set_sliced_input(v_i_param, v, 0, 1, 1, -1, 1);
    loop->set_merged_input(h_param, h0, h_res);
    loop->set_sliced_input(g_i_param, g, 0, 1, 1, -1, 1);
    loop->set_sliced_input(beta_i_param, beta, 0, 1, 1, -1, 1);
    loop->set_merged_input(core_attn_buf, core_attn_init, core_buf_res);
    loop->set_special_body_ports({0, 0});

    auto core_attn_final = loop->get_iter_value(core_buf_res, -1);
    auto h_final = loop->get_iter_value(h_res, -1);

    return std::make_shared<ov::Model>(ov::OutputVector{core_attn_final, h_final},
                                       ov::ParameterVector{q, k, v, h0, g, beta});
}

std::string GatedDeltaNet::getTestCaseName(const testing::TestParamInfo<gated_delta_net_params>& obj) {
    const auto& [batch, seq_len, qk_head_num, v_head_num, head_size, prec, device] = obj.param;
    std::ostringstream result;
    result << "batch=" << batch;
    result << ",seq_len=" << seq_len;
    result << ",qk_head_num=" << qk_head_num;
    result << ",v_head_num=" << v_head_num;
    result << ",head_size=" << head_size;
    result << ",prec=" << prec;
    result << ",device=" << device;
    return result.str();
}

void GatedDeltaNet::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& params = function->get_parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        const auto& param = params[i];
        const auto& shape = targetInputStaticShapes[i];
        if (i == 4) {
            // g: allow negative values
            inputs[param] = ov::test::utils::create_and_fill_tensor(
                param->get_element_type(),
                shape,
                ov::test::utils::InputGenerateData(-0.5, 0.5, 1000, 1));
        } else if (i == 5) {
            // beta: values in [0, 1]
            inputs[param] = ov::test::utils::create_and_fill_tensor(
                param->get_element_type(),
                shape,
                ov::test::utils::InputGenerateData(0.0, 1, 1000, 1));
        } else {
            inputs[param] = ov::test::utils::create_and_fill_tensor(param->get_element_type(), shape);
        }
    }
}

void GatedDeltaNet::compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        ov::test::utils::compare(expected[i], actual[i], abs_threshold, rel_threshold);
    }
}

void GatedDeltaNet::SetUp() {
    const auto& [batch, seq_len, qk_head_num, v_head_num, head_size, prec, device] = GetParam();

    targetDevice = device;
    inType = prec;
    configuration[ov::hint::inference_precision.name()] = prec;

    const ov::Shape q_shape{static_cast<size_t>(batch),
                            static_cast<size_t>(seq_len),
                            static_cast<size_t>(qk_head_num),
                            static_cast<size_t>(head_size)};
    const ov::Shape v_shape{static_cast<size_t>(batch),
                            static_cast<size_t>(seq_len),
                            static_cast<size_t>(v_head_num),
                            static_cast<size_t>(head_size)};
    const ov::Shape h_shape{static_cast<size_t>(batch),
                            static_cast<size_t>(qk_head_num),
                            static_cast<size_t>(head_size),
                            static_cast<size_t>(head_size)};
    const ov::Shape g_shape{static_cast<size_t>(batch),
                            static_cast<size_t>(seq_len),
                            static_cast<size_t>(qk_head_num)};

    init_input_shapes(static_shapes_to_test_representation({q_shape, q_shape, v_shape, h_shape, g_shape, g_shape}));

    auto q = std::make_shared<ov::op::v0::Parameter>(prec, q_shape);
    auto k = std::make_shared<ov::op::v0::Parameter>(prec, q_shape);
    auto v = std::make_shared<ov::op::v0::Parameter>(prec, v_shape);
    auto h0 = std::make_shared<ov::op::v0::Parameter>(prec, h_shape);
    auto g = std::make_shared<ov::op::v0::Parameter>(prec, g_shape);
    auto beta = std::make_shared<ov::op::v0::Parameter>(prec, g_shape);

    q->set_friendly_name("q");
    k->set_friendly_name("k");
    v->set_friendly_name("v");
    h0->set_friendly_name("h0");
    g->set_friendly_name("g");
    beta->set_friendly_name("beta");

    // auto l2norm = [&](const ov::Output<ov::Node>& x) {
    //     auto sq = std::make_shared<ov::op::v1::Multiply>(x, x);
    //     auto axis = ov::op::v0::Constant::create(ov::element::i32, {1}, {3});
    //     auto sum = std::make_shared<ov::op::v1::ReduceSum>(sq, axis, true);
    //     auto eps = ov::op::v0::Constant::create(prec, {}, {1e-6f});
    //     auto inv = std::make_shared<ov::op::v1::Divide>(
    //         ov::op::v0::Constant::create(prec, {}, {1.0f}),
    //         std::make_shared<ov::op::v0::Sqrt>(std::make_shared<ov::op::v1::Add>(sum, eps)));
    //     return std::make_shared<ov::op::v1::Multiply>(x, inv);
    // };

    // auto q_norm = l2norm(q);
    // auto k_norm = l2norm(k);
    // auto q_scale = ov::op::v0::Constant::create(prec, {}, {1.0f / std::sqrt(static_cast<float>(head_size))});
    // auto q_scaled = std::make_shared<ov::op::v1::Multiply>(q_norm, q_scale);

    auto gdn = std::make_shared<ov::op::GatedDeltaNet>(ov::OutputVector{q, k, v, h0, g, beta});
    gdn->set_config({true, true});
    function = std::make_shared<ov::Model>(
        ov::OutputVector{gdn->output(0), gdn->output(1)},
        ov::ParameterVector{q, k, v, h0, g, beta},
        "GatedDeltaNet");

    functionRefs = buildLoopedGDN(batch, seq_len, qk_head_num, v_head_num, head_size);
    ov::serialize(functionRefs, "loop_gdn.xml");
}

void GatedDeltaNet::TearDown() {
}

}  // namespace test
}  // namespace ov