// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Tests that verify reference decompositions stored in
// `src/common/decompositions/` are always recognised and folded back into
// their corresponding internal fused op by the matching transformation.

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/decompositions/rope.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"

using namespace testing;
using namespace ov::pass;

TEST_F(TransformationTestsF, DecompositionRope_FusedByRoPEFusion) {
    disable_rt_info_check();

    const size_t batch = 2;
    const size_t num_heads = 4;
    const size_t seq_length = 16;
    const size_t head_size = 64;
    const int64_t half_head_size = static_cast<int64_t>(head_size / 2);

    const ov::Shape x_shape{batch, num_heads, seq_length, head_size};
    const ov::Shape cos_sin_shape{batch, 1, seq_length, head_size / 2};

    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, x_shape);
        auto cos = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, cos_sin_shape);
        auto sin = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, cos_sin_shape);

        ov::pass::NodeRegistry reg;
        auto rope = ov::decompositions::rope(reg, x, cos, sin, half_head_size);

        model = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{x, cos, sin});
        manager.register_pass<RoPEFusion>();
    }
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, x_shape);
        auto cos = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, cos_sin_shape);
        auto sin = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, cos_sin_shape);

        ov::op::internal::RoPE::Config config;
        config.rotary_ndims = head_size;
        config.cos_sin_ndims = head_size / 2;
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{x, cos, sin}, config);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope}, ov::ParameterVector{x, cos, sin});
    }
}
