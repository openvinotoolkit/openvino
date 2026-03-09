// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/gated_delta_net.hpp"

#include <cmath>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
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
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace test {

std::shared_ptr<ov::Model> GatedDeltaNet::buildLoopedGDN(int32_t batch,
                                                         int32_t seq_len,
                                                         int32_t qk_head_num,
                                                         int32_t v_head_num,
                                                         int32_t head_size) {
    constexpr auto dtype = ov::element::f32;
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
        auto axis = ov::op::v0::Constant::create(ov::element::i32, {1}, {-1});
        auto sum = std::make_shared<ov::op::v1::ReduceSum>(sq, axis, true);
        auto eps = ov::op::v0::Constant::create(dtype, {}, {1e-6f});
        auto inv = std::make_shared<ov::op::v1::Divide>(
            ov::op::v0::Constant::create(dtype, {}, {1.0f}),
            std::make_shared<ov::op::v0::Sqrt>(std::make_shared<ov::op::v1::Add>(sum, eps)));
        return std::make_shared<ov::op::v1::Multiply>(x, inv);
    };

    auto q_norm = l2norm(q);
    auto k_norm = l2norm(k);

    auto v_shape = std::make_shared<ov::op::v3::ShapeOf>(v);
    auto core_attn_init =
        std::make_shared<ov::op::v3::Broadcast>(ov::op::v0::Constant::create(dtype, {}, {0.0f}), v_shape);

    auto perm_bhsd = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});
    auto perm_bhs = ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 2, 1});
    auto q_norm_t = std::make_shared<ov::op::v1::Transpose>(q_norm, perm_bhsd);
    auto q_norm_t_shape = std::make_shared<ov::op::v3::ShapeOf>(q_norm_t);
    auto q_d_index = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    auto q_gather_axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    auto q_d_i64 = std::make_shared<ov::op::v8::Gather>(q_norm_t_shape, q_d_index, q_gather_axis);
    auto q_d = std::make_shared<ov::op::v0::Convert>(q_d_i64, dtype);
    auto half = ov::op::v0::Constant::create(dtype, {}, {0.5f});
    auto q_scale = std::make_shared<ov::op::v1::Power>(q_d, half);
    auto q_scaled_t = std::make_shared<ov::op::v1::Divide>(q_norm_t, q_scale);
    auto k_norm_t = std::make_shared<ov::op::v1::Transpose>(k_norm, perm_bhsd);
    auto v_t = std::make_shared<ov::op::v1::Transpose>(v, perm_bhsd);
    auto g_t = std::make_shared<ov::op::v1::Transpose>(g, perm_bhs);
    auto beta_t = std::make_shared<ov::op::v1::Transpose>(beta, perm_bhs);

    auto vt_shape = std::make_shared<ov::op::v3::ShapeOf>(v_t);
    core_attn_init = std::make_shared<ov::op::v3::Broadcast>(ov::op::v0::Constant::create(dtype, {}, {0.0f}), vt_shape);

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
    loop->set_sliced_input(k_i_param, k_norm_t, 0, 1, 1, -1, 2);
    loop->set_sliced_input(v_i_param, v_t, 0, 1, 1, -1, 2);
    loop->set_sliced_input(g_i_param, g_t, 0, 1, 1, -1, 2);
    loop->set_sliced_input(beta_i_param, beta_t, 0, 1, 1, -1, 2);
    loop->set_merged_input(h_param, h0, h_res);
    loop->set_merged_input(core_attn_buf, core_attn_init, core_buf_res);
    loop->set_special_body_ports({0, 0});

    auto core_attn_final_bhsd = loop->get_iter_value(core_buf_res, -1);
    auto h_final = loop->get_iter_value(h_res, -1);

    auto reshape_m1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    auto flat_core = std::make_shared<ov::op::v1::Reshape>(core_attn_final_bhsd, reshape_m1, false);
    auto flat_h = std::make_shared<ov::op::v1::Reshape>(h_final, reshape_m1, false);
    auto packed_loop_outputs = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{flat_core, flat_h}, 0);

    auto core_shape = std::make_shared<ov::op::v3::ShapeOf>(v_t);
    auto reduce_axis0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    auto core_numel = std::make_shared<ov::op::v1::ReduceProd>(core_shape, reduce_axis0, true);
    auto state_shape = std::make_shared<ov::op::v3::ShapeOf>(h0);
    auto state_numel = std::make_shared<ov::op::v1::ReduceProd>(state_shape, reduce_axis0, true);
    auto state_slice_end = std::make_shared<ov::op::v1::Add>(core_numel, state_numel);
    auto slice_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    auto slice_step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto slice_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});

    auto core_slice =
        std::make_shared<ov::op::v8::Slice>(packed_loop_outputs, slice_start, core_numel, slice_step, slice_axis);
    auto state_slice =
        std::make_shared<ov::op::v8::Slice>(packed_loop_outputs, core_numel, state_slice_end, slice_step, slice_axis);

    auto core_restored = std::make_shared<ov::op::v1::Reshape>(core_slice, core_shape, false);
    auto state_restored = std::make_shared<ov::op::v1::Reshape>(state_slice, state_shape, false);

    auto core_attn_final =
        std::make_shared<ov::op::v1::Transpose>(core_restored,
                                                ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));

    return std::make_shared<ov::Model>(ov::OutputVector{core_attn_final, state_restored},
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
            inputs[param] = ov::test::utils::create_and_fill_tensor(param->get_element_type(),
                                                                    shape,
                                                                    ov::test::utils::InputGenerateData(-1, 1, 1000, 1));
        } else if (i == 5) {
            // beta: values in [0, 1]
            inputs[param] =
                ov::test::utils::create_and_fill_tensor(param->get_element_type(),
                                                        shape,
                                                        ov::test::utils::InputGenerateData(0.0, 1, 1000, 1));
        } else {
            inputs[param] = ov::test::utils::create_and_fill_tensor(param->get_element_type(), shape);
        }
    }
}

void GatedDeltaNet::compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    ov::test::utils::compare(expected[0], actual[0], abs_threshold, rel_threshold);
    ov::test::utils::compare(expected[1], actual[1], abs_threshold, rel_threshold);
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
    const ov::Shape g_shape{static_cast<size_t>(batch), static_cast<size_t>(seq_len), static_cast<size_t>(qk_head_num)};

    init_input_shapes(static_shapes_to_test_representation({q_shape, q_shape, v_shape, h_shape, g_shape, g_shape}));

    function = buildLoopedGDN(batch, seq_len, qk_head_num, v_head_num, head_size);
}

}  // namespace test
}  // namespace ov