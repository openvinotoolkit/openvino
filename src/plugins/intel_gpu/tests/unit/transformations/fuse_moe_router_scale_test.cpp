// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/fuse_moe_router_scale.hpp"

#include <gtest/gtest.h>

#include <limits>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "ov_ops/moe_compressed.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov::test::intel_gpu {

namespace {

using MOECompressed = ov::op::internal::MOECompressed;

// Returns {gate_w, gate_scale, gate_zp, up_w, up_scale, up_zp, down_w, down_scale, down_zp}
ov::OutputVector gemm3_weights(size_t num_experts, size_t hidden_size, size_t inter_size, float down_scale_val = 1.0f) {
    using C = ov::op::v0::Constant;
    const size_t gate_w = num_experts * inter_size * hidden_size;
    const size_t gate_s = num_experts * inter_size;
    const size_t down_w = num_experts * hidden_size * inter_size;
    const size_t down_s = num_experts * hidden_size;
    return {
        C::create(ov::element::u8,  {num_experts, inter_size,  hidden_size}, std::vector<uint8_t>(gate_w, 1)),
        C::create(ov::element::f16, {num_experts, inter_size,  1},           std::vector<float>(gate_s, 1.0f)),
        C::create(ov::element::u8,  {num_experts, inter_size,  1},           std::vector<uint8_t>(gate_s, 0)),
        C::create(ov::element::u8,  {num_experts, inter_size,  hidden_size}, std::vector<uint8_t>(gate_w, 1)),
        C::create(ov::element::f16, {num_experts, inter_size,  1},           std::vector<float>(gate_s, 1.0f)),
        C::create(ov::element::u8,  {num_experts, inter_size,  1},           std::vector<uint8_t>(gate_s, 0)),
        C::create(ov::element::u8,  {num_experts, hidden_size, inter_size},  std::vector<uint8_t>(down_w, 1)),
        C::create(ov::element::f16, {num_experts, hidden_size, 1},           std::vector<float>(down_s, down_scale_val)),
        C::create(ov::element::u8,  {num_experts, hidden_size, 1},           std::vector<uint8_t>(down_s, 0)),
    };
}

MOECompressed::Config gemm3_config(size_t num_experts, size_t hidden_size, size_t inter_size, size_t top_k) {
    MOECompressed::Config c;
    c.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
    c.activation_type = ov::op::internal::MOE::Activation_type::SWIGLU;
    c.hidden_size = hidden_size;
    c.inter_size = inter_size;
    c.num_expert = num_experts;
    c.num_shared_expert = 0;
    c.top_k = top_k;
    c.group_size = std::numeric_limits<size_t>::max();
    c.has_zp = true;
    c.out_type = ov::element::f16;
    return c;
}

}  // namespace

TEST_F(TransformationTestsF, FuseMoEPerExpertScale) {
    const size_t num_experts = 4, hidden_size = 8, inter_size = 4, top_k = 2;
    const std::vector<float> per_expert_scales = {1.0f, 2.0f, 1.0f, 0.5f};

    {
        auto hidden   = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, hidden_size});
        auto routing  = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, top_k});
        auto topk_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, top_k});

        auto pes_const = ov::op::v0::Constant::create(ov::element::f16, {num_experts}, per_expert_scales);
        auto axis      = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto gathered  = std::make_shared<ov::op::v8::Gather>(pes_const, topk_idx, axis);
        auto scaled    = std::make_shared<ov::op::v1::Multiply>(routing, gathered);

        auto ws  = gemm3_weights(num_experts, hidden_size, inter_size);
        auto moe = std::make_shared<MOECompressed>(
            ov::OutputVector{hidden, scaled, topk_idx, ws[0], ws[1], ws[2], ws[3], ws[4], ws[5], ws[6], ws[7], ws[8]},
            gemm3_config(num_experts, hidden_size, inter_size, top_k));

        model = std::make_shared<ov::Model>(ov::OutputVector{moe}, ov::ParameterVector{hidden, routing, topk_idx});
        manager.register_pass<FuseMoERouterScale>();
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    }

    {
        auto hidden   = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, hidden_size});
        auto routing  = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, top_k});
        auto topk_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, top_k});

        std::vector<float> folded_down_scale(num_experts * hidden_size);
        for (size_t n = 0; n < num_experts; ++n)
            for (size_t h = 0; h < hidden_size; ++h)
                folded_down_scale[n * hidden_size + h] = per_expert_scales[n];

        auto ws         = gemm3_weights(num_experts, hidden_size, inter_size);
        auto down_scale = ov::op::v0::Constant::create(ov::element::f16, {num_experts, hidden_size, 1}, folded_down_scale);
        auto moe        = std::make_shared<MOECompressed>(
            ov::OutputVector{hidden, routing, topk_idx, ws[0], ws[1], ws[2], ws[3], ws[4], ws[5], ws[6], down_scale, ws[8]},
            gemm3_config(num_experts, hidden_size, inter_size, top_k));

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{moe}, ov::ParameterVector{hidden, routing, topk_idx});
    }
}

TEST_F(TransformationTestsF, FuseMoEScalarScale) {
    const size_t num_experts = 4, hidden_size = 8, inter_size = 4, top_k = 2;
    const float scale_val = 2.0f;

    {
        auto hidden   = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, hidden_size});
        auto routing  = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, top_k});
        auto topk_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, top_k});

        auto scalar = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {scale_val});
        auto scaled = std::make_shared<ov::op::v1::Multiply>(routing, scalar);

        auto ws  = gemm3_weights(num_experts, hidden_size, inter_size);
        auto moe = std::make_shared<MOECompressed>(
            ov::OutputVector{hidden, scaled, topk_idx, ws[0], ws[1], ws[2], ws[3], ws[4], ws[5], ws[6], ws[7], ws[8]},
            gemm3_config(num_experts, hidden_size, inter_size, top_k));

        model = std::make_shared<ov::Model>(ov::OutputVector{moe}, ov::ParameterVector{hidden, routing, topk_idx});
        manager.register_pass<FuseMoERouterScale>();
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    }

    {
        auto hidden   = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, hidden_size});
        auto routing  = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, top_k});
        auto topk_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, top_k});

        auto ws         = gemm3_weights(num_experts, hidden_size, inter_size);
        auto down_scale = ov::op::v0::Constant::create(ov::element::f16,
                                                        {num_experts, hidden_size, 1},
                                                        std::vector<float>(num_experts * hidden_size, scale_val));
        auto moe        = std::make_shared<MOECompressed>(
            ov::OutputVector{hidden, routing, topk_idx, ws[0], ws[1], ws[2], ws[3], ws[4], ws[5], ws[6], down_scale, ws[8]},
            gemm3_config(num_experts, hidden_size, inter_size, top_k));

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{moe}, ov::ParameterVector{hidden, routing, topk_idx});
    }
}

}  // namespace ov::test::intel_gpu
