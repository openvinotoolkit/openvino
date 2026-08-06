// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/selective_ssm.hpp"

#include <climits>
#include <sstream>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::test {

std::shared_ptr<ov::Model> SelectiveSSM::buildLoopedSelectiveSSM(int32_t num_heads,
                                                                 int32_t num_groups,
                                                                 int32_t head_dim,
                                                                 int32_t state_size,
                                                                 ov::element::Type dtype) {
    using namespace ov::op;

    const int32_t heads_per_group = num_heads / num_groups;
    auto A = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{num_heads});
    auto dt = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, -1, num_heads});
    auto B = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, -1, num_groups, state_size});
    auto x = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, -1, num_heads, head_dim});
    auto C = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, -1, num_groups, state_size});
    auto h0 = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, num_heads, head_dim, state_size});

    auto expand_groups = [&](const std::shared_ptr<v0::Parameter>& src) {
        auto unsq_axis = v0::Constant::create(ov::element::i32, {}, {3});
        auto src_5d = std::make_shared<v0::Unsqueeze>(src, unsq_axis);
        auto tile_shape = v0::Constant::create(ov::element::i64, {5}, {1, 1, 1, heads_per_group, 1});
        auto tiled = std::make_shared<v0::Tile>(src_5d, tile_shape);
        auto target = v0::Constant::create(ov::element::i64, {4}, {0, 0, num_heads, state_size});
        return std::make_shared<v1::Reshape>(tiled, target, true);
    };
    auto B_expanded = expand_groups(B);
    auto C_expanded = expand_groups(C);
    auto dA = std::make_shared<v0::Exp>(std::make_shared<v1::Multiply>(A, dt));
    auto dtB = std::make_shared<v1::Multiply>(std::make_shared<v0::Unsqueeze>(dt, v0::Constant::create(ov::element::i32, {1}, {-1})),
                                              B_expanded);

    auto shape_of_x = std::make_shared<v3::ShapeOf>(x);
    auto core_init = std::make_shared<v3::Broadcast>(v0::Constant::create(dtype, {}, {0.0f}), shape_of_x);
    auto trip_count_i64 = std::make_shared<v8::Gather>(shape_of_x,
                                                       v0::Constant::create(ov::element::i64, {1}, {1}),
                                                       v0::Constant::create(ov::element::i64, {}, {0}));
    auto trip_count = std::make_shared<v0::Convert>(trip_count_i64, ov::element::i32);

    auto timestep = std::make_shared<v0::Parameter>(ov::element::i32, ov::Shape{});
    auto dA_t = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, 1, num_heads});
    auto dtB_t = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, 1, num_heads, state_size});
    auto x_t = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, 1, num_heads, head_dim});
    auto C_t = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, 1, num_heads, state_size});
    auto last_state = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, num_heads, head_dim, state_size});
    auto core_out = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, -1, num_heads, head_dim});

    auto axis1 = v0::Constant::create(ov::element::i32, {}, {1});
    auto minus1 = v0::Constant::create(ov::element::i32, {1}, {-1});
    auto minus2 = v0::Constant::create(ov::element::i32, {1}, {-2});
    auto dA_sq = std::make_shared<v0::Squeeze>(dA_t, axis1);
    auto dtB_sq = std::make_shared<v0::Squeeze>(dtB_t, axis1);
    auto x_sq = std::make_shared<v0::Squeeze>(x_t, axis1);
    auto C_sq = std::make_shared<v0::Squeeze>(C_t, axis1);
    auto dA_4d = std::make_shared<v0::Unsqueeze>(std::make_shared<v0::Unsqueeze>(dA_sq, minus1), minus1);
    auto dBx = std::make_shared<v1::Multiply>(std::make_shared<v0::Unsqueeze>(dtB_sq, minus2),
                                              std::make_shared<v0::Unsqueeze>(x_sq, minus1));
    auto state_new = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(last_state, dA_4d), dBx);
    auto y = std::make_shared<v1::Multiply>(state_new, std::make_shared<v0::Unsqueeze>(C_sq, minus2));
    auto y_sum = std::make_shared<v1::ReduceSum>(y, minus1, false);
    auto y_unsq = std::make_shared<v0::Unsqueeze>(y_sum, axis1);
    auto timestep_unsq = std::make_shared<v0::Unsqueeze>(timestep, v0::Constant::create(ov::element::i32, {1}, {0}));
    auto core_out_new = std::make_shared<v3::ScatterUpdate>(core_out, timestep_unsq, y_unsq, axis1);

    auto body_cond = v0::Constant::create(ov::element::boolean, {1}, {true});
    auto body = std::make_shared<ov::Model>(ov::OutputVector{body_cond, state_new, core_out_new},
                                            ov::ParameterVector{timestep, dA_t, dtB_t, x_t, C_t, last_state, core_out},
                                            "selective_ssm_body");

    auto loop = std::make_shared<v5::Loop>(trip_count, v0::Constant::create(ov::element::boolean, {1}, {true}));
    loop->set_function(body);
    loop->set_sliced_input(dA_t, dA, 0, 1, 1, -1, 1);
    loop->set_sliced_input(dtB_t, dtB, 0, 1, 1, -1, 1);
    loop->set_sliced_input(x_t, x, 0, 1, 1, -1, 1);
    loop->set_sliced_input(C_t, C_expanded, 0, 1, 1, -1, 1);
    loop->set_merged_input(last_state, h0, state_new);
    loop->set_merged_input(core_out, core_init, core_out_new);
    loop->set_special_body_ports({0, 0});

    return std::make_shared<ov::Model>(ov::OutputVector{loop->get_iter_value(core_out_new, -1), loop->get_iter_value(state_new, -1)},
                                       ov::ParameterVector{A, dt, B, x, C, h0});
}

std::string SelectiveSSM::getTestCaseName(const testing::TestParamInfo<selective_ssm_params>& obj) {
    const auto& [batch, seq_len, num_heads, num_groups, head_dim, state_size, prec, device] = obj.param;
    std::ostringstream result;
    result << "batch=" << batch;
    result << ",seq_len=" << seq_len;
    result << ",num_heads=" << num_heads;
    result << ",num_groups=" << num_groups;
    result << ",head_dim=" << head_dim;
    result << ",state_size=" << state_size;
    result << ",prec=" << prec;
    result << ",device=" << device;
    return result.str();
}

void SelectiveSSM::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& params = function->get_parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        const auto& param = params[i];
        const auto& shape = targetInputStaticShapes[i];
        if (i == 0) {
            inputs[param] = ov::test::utils::create_and_fill_tensor(param->get_element_type(),
                                                                    shape,
                                                                    ov::test::utils::InputGenerateData(-0.5f, 0.7f, 1000, 1));
        } else if (i == 1) {
            inputs[param] = ov::test::utils::create_and_fill_tensor(param->get_element_type(),
                                                                    shape,
                                                                    ov::test::utils::InputGenerateData(0.0f, 0.5f, 1000, 1));
        } else {
            inputs[param] = ov::test::utils::create_and_fill_tensor(param->get_element_type(),
                                                                    shape,
                                                                    ov::test::utils::InputGenerateData(-0.5f, 1.0f, 1000, 1));
        }
    }
}

void SelectiveSSM::compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    ov::test::utils::compare(expected[0], actual[0], abs_threshold, rel_threshold);
    ov::test::utils::compare(expected[1], actual[1], abs_threshold, rel_threshold);
}

void SelectiveSSM::SetUp() {
    const auto& [batch, seq_len, num_heads, num_groups, head_dim, state_size, prec, device] = GetParam();

    targetDevice = device;
    inType = prec;
    configuration[ov::hint::inference_precision.name()] = prec;

    abs_threshold = prec == ov::element::f32 ? 1e-6f : 1e-3f;
    rel_threshold = 1e-5f;

    const ov::Shape A_shape{static_cast<size_t>(num_heads)};
    const ov::Shape dt_shape{static_cast<size_t>(batch), static_cast<size_t>(seq_len), static_cast<size_t>(num_heads)};
    const ov::Shape B_shape{static_cast<size_t>(batch), static_cast<size_t>(seq_len), static_cast<size_t>(num_groups), static_cast<size_t>(state_size)};
    const ov::Shape x_shape{static_cast<size_t>(batch), static_cast<size_t>(seq_len), static_cast<size_t>(num_heads), static_cast<size_t>(head_dim)};
    const ov::Shape state_shape{static_cast<size_t>(batch), static_cast<size_t>(num_heads), static_cast<size_t>(head_dim), static_cast<size_t>(state_size)};

    init_input_shapes(static_shapes_to_test_representation({A_shape, dt_shape, B_shape, x_shape, B_shape, state_shape}));
    function = buildLoopedSelectiveSSM(num_heads, num_groups, head_dim, state_size, prec);
}

}  // namespace ov::test
