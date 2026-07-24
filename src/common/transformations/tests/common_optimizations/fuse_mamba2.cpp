// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_mamba2.hpp"

#include <gtest/gtest.h>

#include <climits>
#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/mamba2.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"

namespace ov::test {
namespace {

// Builds a loop-based Mamba2 recurrence model:
//   state_t = state_{t-1} * dA_t + dBx_t ; y_t = reduce_sum(state_t * C_t, axis=state_size)
// External Loop inputs (in order): trip_count, exec_cond, dA, dBx, C, recurrent_state, output_buffer.
// External Loop outputs: output[B,H,L,P], output_recurrent_state[B,H,P,N].
//
// \param with_post_loop appends the flatten/Concat/Slice/Reshape round-trip seen in real models.
// \param break_body replaces the state `Add` with `Subtract` so the body no longer matches Mamba2.
std::shared_ptr<ov::Model> build_looped_mamba2(int32_t num_heads,
                                               int32_t head_dim,
                                               int32_t state_size,
                                               ov::element::Type dtype = ov::element::f32,
                                               bool with_post_loop = false,
                                               bool break_body = false) {
    using namespace ov::op;

    ov::PartialShape dA_shape{-1, num_heads, -1, 1, 1};
    ov::PartialShape dBx_shape{-1, num_heads, -1, head_dim, state_size};
    ov::PartialShape C_shape{-1, num_heads, -1, state_size};
    ov::PartialShape state_shape{-1, num_heads, head_dim, state_size};

    auto dA = std::make_shared<v0::Parameter>(dtype, dA_shape);
    auto dBx = std::make_shared<v0::Parameter>(dtype, dBx_shape);
    auto C = std::make_shared<v0::Parameter>(dtype, C_shape);
    auto h0 = std::make_shared<v0::Parameter>(dtype, state_shape);

    // output accumulator buffer: zeros with shape [B, H, L, P]
    auto shape_of_dBx = std::make_shared<v3::ShapeOf>(dBx);
    auto slice_start = v0::Constant::create(ov::element::i64, {1}, {0});
    auto slice_stop = v0::Constant::create(ov::element::i64, {1}, {4});
    auto slice_step = v0::Constant::create(ov::element::i64, {1}, {1});
    auto slice_axis = v0::Constant::create(ov::element::i64, {1}, {0});
    auto out_buffer_shape = std::make_shared<v8::Slice>(shape_of_dBx, slice_start, slice_stop, slice_step, slice_axis);
    auto core_init = std::make_shared<v3::Broadcast>(v0::Constant::create(dtype, {}, {0.0f}), out_buffer_shape);

    // trip count = seq_len (dim 2 of dBx)
    auto trip_index = v0::Constant::create(ov::element::i64, {1}, {2});
    auto trip_axis = v0::Constant::create(ov::element::i64, {}, {0});
    auto trip_count_i64 = std::make_shared<v8::Gather>(shape_of_dBx, trip_index, trip_axis);
    auto trip_count = std::make_shared<v0::Convert>(trip_count_i64, ov::element::i32);

    // -------- Loop body --------
    auto timestep = std::make_shared<v0::Parameter>(ov::element::i32, ov::Shape{});
    auto dA_t = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, num_heads, 1, 1, 1});
    auto dBx_t = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, num_heads, 1, head_dim, state_size});
    auto C_t = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, num_heads, 1, state_size});
    auto last_state = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, num_heads, head_dim, state_size});
    auto core_out = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, num_heads, -1, head_dim});

    auto axis2 = v0::Constant::create(ov::element::i32, {}, {2});
    auto minus1 = v0::Constant::create(ov::element::i32, {1}, {-1});
    auto minus2 = v0::Constant::create(ov::element::i32, {1}, {-2});

    auto dA_sq = std::make_shared<v0::Squeeze>(dA_t, axis2);
    auto dBx_sq = std::make_shared<v0::Squeeze>(dBx_t, axis2);
    auto C_sq = std::make_shared<v0::Squeeze>(C_t, axis2);

    auto state_decay = std::make_shared<v1::Multiply>(last_state, dA_sq);
    std::shared_ptr<ov::Node> state_new;
    if (break_body) {
        state_new = std::make_shared<v1::Subtract>(state_decay, dBx_sq);
    } else {
        state_new = std::make_shared<v1::Add>(state_decay, dBx_sq);
    }

    auto C_unsq = std::make_shared<v0::Unsqueeze>(C_sq, minus2);
    auto y = std::make_shared<v1::Multiply>(state_new, C_unsq);
    auto y_sum = std::make_shared<v1::ReduceSum>(y, minus1, false);
    auto y_unsq = std::make_shared<v0::Unsqueeze>(y_sum, axis2);

    auto timestep_unsq = std::make_shared<v0::Unsqueeze>(timestep, v0::Constant::create(ov::element::i32, {1}, {0}));
    auto core_out_new = std::make_shared<v3::ScatterUpdate>(core_out, timestep_unsq, y_unsq, axis2);

    auto body_cond = v0::Constant::create(ov::element::boolean, {1}, {true});
    auto body = std::make_shared<ov::Model>(ov::OutputVector{body_cond, state_new, core_out_new},
                                            ov::ParameterVector{timestep, dA_t, dBx_t, C_t, last_state, core_out},
                                            "mamba2_body");

    // -------- Loop --------
    auto loop = std::make_shared<v5::Loop>(trip_count, v0::Constant::create(ov::element::boolean, {1}, {true}));
    loop->set_function(body);
    loop->set_sliced_input(dA_t, dA, 0, 1, 1, -1, 2);
    loop->set_sliced_input(dBx_t, dBx, 0, 1, 1, -1, 2);
    loop->set_sliced_input(C_t, C, 0, 1, 1, -1, 2);
    loop->set_merged_input(last_state, h0, state_new);
    loop->set_merged_input(core_out, core_init, core_out_new);
    loop->set_special_body_ports({0, 0});

    auto output = loop->get_iter_value(core_out_new, -1);  // Loop output(0)
    auto state_out = loop->get_iter_value(state_new, -1);  // Loop output(1)

    ov::Output<ov::Node> final_output = output;
    ov::Output<ov::Node> final_state = state_out;

    if (with_post_loop) {
        auto reshape_m1 = v0::Constant::create(ov::element::i64, {1}, {-1});
        auto flat_out = std::make_shared<v1::Reshape>(output, reshape_m1, false);
        auto flat_state = std::make_shared<v1::Reshape>(state_out, reshape_m1, false);
        auto packed = std::make_shared<v0::Concat>(ov::OutputVector{flat_out, flat_state}, 0);

        auto out_shape = std::make_shared<v3::ShapeOf>(core_init);
        auto state_ref_shape = std::make_shared<v3::ShapeOf>(h0);
        auto reduce_axis0 = v0::Constant::create(ov::element::i64, {1}, {0});
        auto out_numel = std::make_shared<v1::ReduceProd>(out_shape, reduce_axis0, true);

        auto s_start = v0::Constant::create(ov::element::i64, {1}, {0});
        auto s_step = v0::Constant::create(ov::element::i64, {1}, {1});
        auto s_axis = v0::Constant::create(ov::element::i64, {1}, {0});
        auto s_end_inf = v0::Constant::create(ov::element::i64, {1}, {LLONG_MAX});

        auto out_slice = std::make_shared<v8::Slice>(packed, s_start, out_numel, s_step, s_axis);
        auto state_slice = std::make_shared<v8::Slice>(packed, out_numel, s_end_inf, s_step, s_axis);
        final_output = std::make_shared<v1::Reshape>(out_slice, out_shape, false);
        final_state = std::make_shared<v1::Reshape>(state_slice, state_ref_shape, false);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{final_output, final_state},
                                       ov::ParameterVector{dA, dBx, C, h0});
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

TEST(TransformationTests, Mamba2Fusion_FuseLoop) {
    auto model = build_looped_mamba2(/*num_heads=*/4, /*head_dim=*/8, /*state_size=*/16);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Mamba2Fusion>();
    manager.run_passes(model);

    EXPECT_EQ(count_ops_of_type(model, "Loop"), 0u);
    EXPECT_EQ(count_ops_of_type(model, "Mamba2"), 1u);
}

TEST(TransformationTests, Mamba2Fusion_FuseLoopWithPostLoopReshape) {
    auto model = build_looped_mamba2(/*num_heads=*/4,
                                     /*head_dim=*/8,
                                     /*state_size=*/16,
                                     ov::element::f32,
                                     /*with_post_loop=*/true);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Mamba2Fusion>();
    manager.run_passes(model);

    EXPECT_EQ(count_ops_of_type(model, "Loop"), 0u);
    EXPECT_EQ(count_ops_of_type(model, "Mamba2"), 1u);
}

TEST(TransformationTests, Mamba2Fusion_DoesNotFuseOnBrokenBody) {
    auto model = build_looped_mamba2(/*num_heads=*/4,
                                     /*head_dim=*/8,
                                     /*state_size=*/16,
                                     ov::element::f32,
                                     /*with_post_loop=*/false,
                                     /*break_body=*/true);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Mamba2Fusion>();
    manager.run_passes(model);

    // Body does not match the Mamba2 recurrence, so the Loop must be preserved.
    EXPECT_EQ(count_ops_of_type(model, "Mamba2"), 0u);
    EXPECT_EQ(count_ops_of_type(model, "Loop"), 1u);
}

}  // namespace ov::test
