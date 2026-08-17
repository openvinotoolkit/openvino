// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_ssm.hpp"

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
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace ov::test {
namespace {

// Builds a loop-based SSM recurrence model matching the time-major export where all
// discretization is performed outside the loop and consumed as 5D per-step slices:
//   dA  = reshape(exp(A * dt), [B, T, H, 1, 1])                       (Loop input 2)
//   dBx = unsqueeze(dt*B, -2) * unsqueeze(x, -1)  -> [B, T, H, P, N]  (Loop input 3)
//   C   = unsqueeze(C_expanded, -2)               -> [B, T, H, 1, N]  (Loop input 4)
// Per step: state_t = state_{t-1} * dA_t + dBx_t; y_t = reduce_sum(state_t * C_t, axis=state_size).
// B/C are provided per group and expanded to heads via Unsqueeze/Tile/Reshape outside the loop.
// External Loop inputs (in order): trip_count, exec_cond, dA, dBx, C, recurrent_state, output_buffer.
// External Loop outputs: output[B,T,H,P], output_recurrent_state[B,H,P,N].
//
// \param with_post_loop appends the flatten/Concat/Slice/Reshape round-trip seen in real models.
// \param break_body replaces the state `Add` with `Subtract` so the body no longer matches the SSM recurrence.
std::shared_ptr<ov::Model> build_looped_ssm(int32_t num_heads,
                                            int32_t num_groups,
                                            int32_t head_dim,
                                            int32_t state_size,
                                            bool with_post_loop = false,
                                            bool break_body = false) {
    using namespace ov::op;

    const auto dtype = ov::element::f32;
    const int32_t heads_per_group = num_heads / num_groups;

    ov::PartialShape dt_shape{-1, -1, num_heads};
    ov::PartialShape B_shape{-1, -1, num_groups, state_size};
    ov::PartialShape x_shape{-1, -1, num_heads, head_dim};
    ov::PartialShape C_shape{-1, -1, num_groups, state_size};
    ov::PartialShape state_shape{-1, num_heads, head_dim, state_size};

    auto dt = std::make_shared<v0::Parameter>(dtype, dt_shape);
    auto B = std::make_shared<v0::Parameter>(dtype, B_shape);
    auto x = std::make_shared<v0::Parameter>(dtype, x_shape);
    auto C = std::make_shared<v0::Parameter>(dtype, C_shape);
    auto h0 = std::make_shared<v0::Parameter>(dtype, state_shape);

    // A is a per-head constant embedded in the graph (log-decay rates).
    auto A = v0::Constant::create(dtype, {static_cast<size_t>(num_heads)}, std::vector<float>(num_heads, -0.5f));

    // Expand B/C from [B, T, G, N] to [B, T, H, N] via Unsqueeze -> Tile -> Reshape.
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

    // Discretization is performed outside the loop and materialized as 5D per-step tensors:
    //   dA  = reshape(exp(A * dt), [B, T, H, 1, 1])
    //   dBx = unsqueeze(unsqueeze(dt, -1) * B_expanded, -2) * unsqueeze(x, -1) -> [B, T, H, P, N]
    //   C   = unsqueeze(C_expanded, -2)                                        -> [B, T, H, 1, N]
    auto minus1_outer = v0::Constant::create(ov::element::i32, {1}, {-1});
    auto minus2_outer = v0::Constant::create(ov::element::i32, {1}, {-2});
    auto dA_shape = v0::Constant::create(ov::element::i64, {5}, {0, 0, 0, 1, 1});
    auto dA_outer =
        std::make_shared<v1::Reshape>(std::make_shared<v0::Exp>(std::make_shared<v1::Multiply>(A, dt)), dA_shape, true);
    auto dB_outer = std::make_shared<v1::Multiply>(std::make_shared<v0::Unsqueeze>(dt, minus1_outer), B_expanded);
    auto dBx_outer = std::make_shared<v1::Multiply>(std::make_shared<v0::Unsqueeze>(dB_outer, minus2_outer),
                                                    std::make_shared<v0::Unsqueeze>(x, minus1_outer));
    auto C_5d_outer = std::make_shared<v0::Unsqueeze>(C_expanded, minus2_outer);

    // output accumulator buffer: zeros with shape [B, T, H, P]
    auto shape_of_x = std::make_shared<v3::ShapeOf>(x);
    auto core_init = std::make_shared<v3::Broadcast>(v0::Constant::create(dtype, {}, {0.0f}), shape_of_x);

    // trip count = seq_len (dim 1 of x)
    auto trip_index = v0::Constant::create(ov::element::i64, {1}, {1});
    auto trip_axis = v0::Constant::create(ov::element::i64, {}, {0});
    auto trip_count_i64 = std::make_shared<v8::Gather>(shape_of_x, trip_index, trip_axis);
    auto trip_count = std::make_shared<v0::Convert>(trip_count_i64, ov::element::i32);

    // -------- Loop body --------
    auto timestep = std::make_shared<v0::Parameter>(ov::element::i32, ov::Shape{});
    auto dA_t = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, 1, num_heads, 1, 1});
    auto dBx_t = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, 1, num_heads, head_dim, state_size});
    auto C_t = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, 1, num_heads, 1, state_size});
    auto last_state = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, num_heads, head_dim, state_size});
    auto core_out = std::make_shared<v0::Parameter>(dtype, ov::PartialShape{-1, -1, num_heads, head_dim});

    auto axis1 = v0::Constant::create(ov::element::i32, {}, {1});
    auto minus1 = v0::Constant::create(ov::element::i32, {1}, {-1});

    auto dA_sq = std::make_shared<v0::Squeeze>(dA_t, axis1);    // [B, H, 1, 1]
    auto dBx_sq = std::make_shared<v0::Squeeze>(dBx_t, axis1);  // [B, H, P, N]
    auto C_sq = std::make_shared<v0::Squeeze>(C_t, axis1);      // [B, H, 1, N]

    auto state_decay = std::make_shared<v1::Multiply>(last_state, dA_sq);
    std::shared_ptr<ov::Node> state_new;
    if (break_body) {
        state_new = std::make_shared<v1::Subtract>(state_decay, dBx_sq);
    } else {
        state_new = std::make_shared<v1::Add>(state_decay, dBx_sq);
    }

    auto y = std::make_shared<v1::Multiply>(state_new, C_sq);
    auto y_sum = std::make_shared<v1::ReduceSum>(y, minus1, false);
    auto y_unsq = std::make_shared<v0::Unsqueeze>(y_sum, axis1);

    auto timestep_unsq = std::make_shared<v0::Unsqueeze>(timestep, v0::Constant::create(ov::element::i32, {1}, {0}));
    auto core_out_new = std::make_shared<v3::ScatterUpdate>(core_out, timestep_unsq, y_unsq, axis1);

    auto body_cond = v0::Constant::create(ov::element::boolean, {1}, {true});
    auto body = std::make_shared<ov::Model>(ov::OutputVector{body_cond, state_new, core_out_new},
                                            ov::ParameterVector{timestep, dA_t, dBx_t, C_t, last_state, core_out},
                                            "ssm_body");

    // -------- Loop --------
    auto loop = std::make_shared<v5::Loop>(trip_count, v0::Constant::create(ov::element::boolean, {1}, {true}));
    loop->set_function(body);
    loop->set_sliced_input(dA_t, dA_outer, 0, 1, 1, -1, 1);
    loop->set_sliced_input(dBx_t, dBx_outer, 0, 1, 1, -1, 1);
    loop->set_sliced_input(C_t, C_5d_outer, 0, 1, 1, -1, 1);
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
                                       ov::ParameterVector{dt, B, x, C, h0});
}

std::shared_ptr<ov::Model> build_fused_ssm(int32_t num_heads,
                                           int32_t num_groups,
                                           int32_t head_dim,
                                           int32_t state_size) {
    using namespace ov::op;

    const auto dtype = ov::element::f32;
    ov::PartialShape dt_shape{-1, -1, num_heads};
    ov::PartialShape B_shape{-1, -1, num_groups, state_size};
    ov::PartialShape x_shape{-1, -1, num_heads, head_dim};
    ov::PartialShape C_shape{-1, -1, num_groups, state_size};
    ov::PartialShape state_shape{-1, num_heads, head_dim, state_size};

    auto dt = std::make_shared<v0::Parameter>(dtype, dt_shape);
    auto B = std::make_shared<v0::Parameter>(dtype, B_shape);
    auto x = std::make_shared<v0::Parameter>(dtype, x_shape);
    auto C = std::make_shared<v0::Parameter>(dtype, C_shape);
    auto h0 = std::make_shared<v0::Parameter>(dtype, state_shape);
    auto A = v0::Constant::create(dtype, {static_cast<size_t>(num_heads)}, std::vector<float>(num_heads, -0.5f));

    auto ssm = std::make_shared<ov::op::internal::SelectiveSSM>(ov::OutputVector{A, dt, B, x, C, h0});

    return std::make_shared<ov::Model>(ov::OutputVector{ssm->output(0), ssm->output(1)},
                                       ov::ParameterVector{dt, B, x, C, h0});
}

}  // namespace

TEST_F(TransformationTestsF, SSMFusion_FuseLoop) {
    model = build_looped_ssm(/*num_heads=*/4, /*num_groups=*/2, /*head_dim=*/8, /*state_size=*/16);
    model_ref = build_fused_ssm(/*num_heads=*/4, /*num_groups=*/2, /*head_dim=*/8, /*state_size=*/16);
    manager.register_pass<ov::pass::SSMFusion>();
}

TEST_F(TransformationTestsF, SSMFusion_FuseLoopWithPostLoopReshape) {
    model = build_looped_ssm(/*num_heads=*/4,
                             /*num_groups=*/2,
                             /*head_dim=*/8,
                             /*state_size=*/16,
                             /*with_post_loop=*/true);
    model_ref = build_fused_ssm(/*num_heads=*/4, /*num_groups=*/2, /*head_dim=*/8, /*state_size=*/16);
    manager.register_pass<ov::pass::SSMFusion>();
    // Removing the post-loop reshape reconnects the original Result to a different producer,
    // so its friendly name differs from the reference.
    disable_result_friendly_names_check();
}

TEST_F(TransformationTestsF, SSMFusion_DoesNotFuseOnBrokenBody) {
    model = build_looped_ssm(/*num_heads=*/4,
                             /*num_groups=*/2,
                             /*head_dim=*/8,
                             /*state_size=*/16,
                             /*with_post_loop=*/false,
                             /*break_body=*/true);
    manager.register_pass<ov::pass::SSMFusion>();
}

}  // namespace ov::test
