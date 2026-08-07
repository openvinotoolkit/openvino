// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_selective_ssm.hpp"

#include <gtest/gtest.h>

#include <climits>
#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
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
#include "openvino/op/subtract.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"

namespace ov::test {
namespace {

std::shared_ptr<ov::Model> build_looped_selective_ssm(int32_t num_heads,
                                                      int32_t num_groups,
                                                      int32_t head_dim,
                                                      int32_t state_size,
                                                      ov::element::Type dtype = ov::element::f32,
                                                      bool with_post_loop = false,
                                                      bool break_body = false) {
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
    auto state_decay = std::make_shared<v1::Multiply>(last_state, dA_4d);
    std::shared_ptr<ov::Node> state_new = break_body ? std::static_pointer_cast<ov::Node>(std::make_shared<v1::Subtract>(state_decay, dBx))
                                                     : std::static_pointer_cast<ov::Node>(std::make_shared<v1::Add>(state_decay, dBx));
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

    auto output = loop->get_iter_value(core_out_new, -1);
    auto state_out = loop->get_iter_value(state_new, -1);
    ov::Output<ov::Node> final_output = output;
    ov::Output<ov::Node> final_state = state_out;
    if (with_post_loop) {
        auto reshape_m1 = v0::Constant::create(ov::element::i64, {1}, {-1});
        auto flat_out = std::make_shared<v1::Reshape>(output, reshape_m1, false);
        auto flat_state = std::make_shared<v1::Reshape>(state_out, reshape_m1, false);
        auto packed = std::make_shared<v0::Concat>(ov::OutputVector{flat_out, flat_state}, 0);
        auto out_shape = std::make_shared<v3::ShapeOf>(core_init);
        auto state_shape = std::make_shared<v3::ShapeOf>(h0);
        auto out_numel = std::make_shared<v1::ReduceProd>(out_shape, v0::Constant::create(ov::element::i64, {1}, {0}), true);
        auto s_start = v0::Constant::create(ov::element::i64, {1}, {0});
        auto s_step = v0::Constant::create(ov::element::i64, {1}, {1});
        auto s_axis = v0::Constant::create(ov::element::i64, {1}, {0});
        auto s_end_inf = v0::Constant::create(ov::element::i64, {1}, {LLONG_MAX});
        auto out_slice = std::make_shared<v8::Slice>(packed, s_start, out_numel, s_step, s_axis);
        auto state_slice = std::make_shared<v8::Slice>(packed, out_numel, s_end_inf, s_step, s_axis);
        final_output = std::make_shared<v1::Reshape>(out_slice, out_shape, false);
        final_state = std::make_shared<v1::Reshape>(state_slice, state_shape, false);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{final_output, final_state},
                                       ov::ParameterVector{A, dt, B, x, C, h0});
}

size_t count_ops_of_type(const std::shared_ptr<ov::Model>& model, const std::string& type_name) {
    size_t count = 0;
    for (const auto& node : model->get_ops()) {
        if (node->get_type_name() == type_name) {
            ++count;
        }
    }
    return count;
}

}  // namespace

TEST(TransformationTests, SelectiveSSMFusion_FuseLoop) {
    auto model = build_looped_selective_ssm(4, 2, 8, 16);
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::SelectiveSSMFusion>();
    manager.run_passes(model);
    EXPECT_EQ(count_ops_of_type(model, "Loop"), 0u);
    EXPECT_EQ(count_ops_of_type(model, "SelectiveSSM"), 1u);
}

TEST(TransformationTests, SelectiveSSMFusion_FuseLoopWithPostLoopReshape) {
    auto model = build_looped_selective_ssm(4, 2, 8, 16, ov::element::f32, true);
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::SelectiveSSMFusion>();
    manager.run_passes(model);
    EXPECT_EQ(count_ops_of_type(model, "Loop"), 0u);
    EXPECT_EQ(count_ops_of_type(model, "SelectiveSSM"), 1u);
}

TEST(TransformationTests, SelectiveSSMFusion_DoesNotFuseOnBrokenBody) {
    auto model = build_looped_selective_ssm(4, 2, 8, 16, ov::element::f32, false, true);
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::SelectiveSSMFusion>();
    manager.run_passes(model);
    EXPECT_EQ(count_ops_of_type(model, "SelectiveSSM"), 0u);
    EXPECT_EQ(count_ops_of_type(model, "Loop"), 1u);
}

}  // namespace ov::test
