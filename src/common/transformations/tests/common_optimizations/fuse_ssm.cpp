// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_ssm.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "openvino/op/util/variable.hpp"
#include "ssm_test_models.hpp"

namespace ov::test {
namespace {

using ov::test::ssm::build_looped_ssm;

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
    auto A = v0::Constant::create(dtype, {static_cast<size_t>(num_heads)}, std::vector<float>(num_heads, -0.5f));

    const auto state_src = ssm::make_recurrent_state_source(state_shape, /*plain_parameter_state=*/false);
    auto ssm = std::make_shared<ov::op::internal::SelectiveSSM>(ov::OutputVector{A, dt, B, x, C, state_src.state});

    ov::ParameterVector params{dt, B, x, C};
    params.insert(params.end(), state_src.params.begin(), state_src.params.end());

    auto assign = std::make_shared<v6::Assign>(ssm->output(1), state_src.variable);
    auto result = std::make_shared<v0::Result>(ssm->output(0));
    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       ov::SinkVector{assign},
                                       params,
                                       ov::op::util::VariableVector{state_src.variable});
}

}  // namespace

TEST_F(TransformationTestsF, SelectiveSSMFusion_FuseLoop) {
    model = build_looped_ssm(/*num_heads=*/4, /*num_groups=*/2, /*head_dim=*/8, /*state_size=*/16);
    model_ref = build_fused_ssm(/*num_heads=*/4, /*num_groups=*/2, /*head_dim=*/8, /*state_size=*/16);
    manager.register_pass<ov::pass::SelectiveSSMFusion>();
}

TEST_F(TransformationTestsF, SelectiveSSMFusion_FuseLoopWithPostLoopReshape) {
    model = build_looped_ssm(/*num_heads=*/4,
                             /*num_groups=*/2,
                             /*head_dim=*/8,
                             /*state_size=*/16,
                             /*with_post_loop=*/true);
    model_ref = build_fused_ssm(/*num_heads=*/4, /*num_groups=*/2, /*head_dim=*/8, /*state_size=*/16);
    manager.register_pass<ov::pass::SelectiveSSMFusion>();
    // Removing the post-loop reshape reconnects the original Result to a different producer,
    // so its friendly name differs from the reference.
    disable_result_friendly_names_check();
}

TEST_F(TransformationTestsF, SelectiveSSMFusion_DoesNotFuseOnBrokenBody) {
    model = build_looped_ssm(/*num_heads=*/4,
                             /*num_groups=*/2,
                             /*head_dim=*/8,
                             /*state_size=*/16,
                             /*with_post_loop=*/false,
                             /*break_body=*/true);
    manager.register_pass<ov::pass::SelectiveSSMFusion>();
}

}  // namespace ov::test
