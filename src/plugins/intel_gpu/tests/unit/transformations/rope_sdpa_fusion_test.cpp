// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "intel_gpu/op/indirect_sdpa.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "plugin/transformations/rope_sdpa_fusion.hpp"

#include <memory>
#include <vector>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

namespace {

constexpr int64_t kBatch = 1;
constexpr int64_t kTokens = 2;
constexpr int64_t kHeads = 4;
constexpr int64_t kHeadSize = 8;

const std::vector<int64_t> kIdentity{0, 1, 2, 3};

ov::op::internal::RoPE::Config rotate_half_config() {
    ov::op::internal::RoPE::Config config;
    config.is_interleaved = true;
    config.rotary_ndims = kHeadSize;
    config.head_cnt = kHeads;
    config.head_size = kHeadSize;
    return config;
}

std::shared_ptr<ov::op::v0::Parameter> q_param(const ov::element::Type& type = ov::element::f16,
                                               const ov::PartialShape& shape = {kBatch,
                                                                                kTokens,
                                                                                kHeads,
                                                                                kHeadSize}) {
    return std::make_shared<ov::op::v0::Parameter>(type, shape);
}

std::shared_ptr<ov::op::v0::Parameter> table_param(const ov::element::Type& type = ov::element::f16) {
    return std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape{kBatch, kTokens, kHeadSize});
}

std::shared_ptr<ov::intel_gpu::op::SDPA> plain_sdpa(const ov::OutputVector& inputs) {
    return std::make_shared<ov::intel_gpu::op::SDPA>(inputs,
                                                     false,
                                                     kIdentity,
                                                     kIdentity,
                                                     kIdentity,
                                                     kIdentity,
                                                     ov::element::dynamic);
}

std::shared_ptr<ov::intel_gpu::op::SDPA> rope_sdpa(const ov::OutputVector& inputs) {
    return std::make_shared<ov::intel_gpu::op::SDPA>(inputs,
                                                     false,
                                                     kIdentity,
                                                     kIdentity,
                                                     kIdentity,
                                                     kIdentity,
                                                     ov::element::dynamic,
                                                     true);
}

}  // namespace

TEST_F(TransformationTestsF, RoPESDPAFusion_QSideFolded) {
    {
        auto q = q_param();
        auto k = q_param();
        auto v = q_param();
        auto cos = table_param();
        auto sin = table_param();
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, rotate_half_config());

        model = std::make_shared<ov::Model>(ov::OutputVector{plain_sdpa({rope, k, v})},
                                            ov::ParameterVector{q, k, v, cos, sin});
        manager.register_pass<RoPESDPAFusion>();
    }
    {
        auto q = q_param();
        auto k = q_param();
        auto v = q_param();
        auto cos = table_param();
        auto sin = table_param();

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope_sdpa({q, k, v, cos, sin})},
                                                ov::ParameterVector{q, k, v, cos, sin});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

// The cos/sin pair goes after the attention mask so the existing input slots keep their indices.
TEST_F(TransformationTestsF, RoPESDPAFusion_QSideFoldedBehindAttentionMask) {
    {
        auto q = q_param();
        auto k = q_param();
        auto v = q_param();
        auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                            ov::PartialShape{kBatch, 1, kTokens, kTokens});
        auto cos = table_param();
        auto sin = table_param();
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, rotate_half_config());

        model = std::make_shared<ov::Model>(ov::OutputVector{plain_sdpa({rope, k, v, mask})},
                                            ov::ParameterVector{q, k, v, mask, cos, sin});
        manager.register_pass<RoPESDPAFusion>();
    }
    {
        auto q = q_param();
        auto k = q_param();
        auto v = q_param();
        auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                            ov::PartialShape{kBatch, 1, kTokens, kTokens});
        auto cos = table_param();
        auto sin = table_param();

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rope_sdpa({q, k, v, mask, cos, sin})},
                                                ov::ParameterVector{q, k, v, mask, cos, sin});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

// Only Q is rotated inside the tile load, so a RoPE on the K input must be left alone.
TEST_F(TransformationTestsF, RoPESDPAFusion_KSideNotFolded) {
    auto q = q_param();
    auto k = q_param();
    auto v = q_param();
    auto cos = table_param();
    auto sin = table_param();
    auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{k, cos, sin}, rotate_half_config());

    model = std::make_shared<ov::Model>(ov::OutputVector{plain_sdpa({q, rope, v})},
                                        ov::ParameterVector{q, k, v, cos, sin});
    manager.register_pass<RoPESDPAFusion>();
}

// A second consumer would still need the rotated Q as a tensor, so the RoPE cannot be deleted.
TEST_F(TransformationTestsF, RoPESDPAFusion_SharedRoPENotFolded) {
    auto q = q_param();
    auto k = q_param();
    auto v = q_param();
    auto cos = table_param();
    auto sin = table_param();
    auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, rotate_half_config());

    model = std::make_shared<ov::Model>(ov::OutputVector{plain_sdpa({rope, k, v}), rope},
                                        ov::ParameterVector{q, k, v, cos, sin});
    manager.register_pass<RoPESDPAFusion>();
}

// The fused rotation in the micro-kernel reads f16 only.
TEST_F(TransformationTestsF, RoPESDPAFusion_F32NotFolded) {
    auto q = q_param(ov::element::f32);
    auto k = q_param(ov::element::f32);
    auto v = q_param(ov::element::f32);
    auto cos = table_param(ov::element::f32);
    auto sin = table_param(ov::element::f32);
    auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, rotate_half_config());

    model = std::make_shared<ov::Model>(ov::OutputVector{plain_sdpa({rope, k, v})},
                                        ov::ParameterVector{q, k, v, cos, sin});
    manager.register_pass<RoPESDPAFusion>();
}

// The kernel indexes the table with constant batch and token counts.
TEST_F(TransformationTestsF, RoPESDPAFusion_DynamicTokensNotFolded) {
    auto q = q_param(ov::element::f16, ov::PartialShape{kBatch, -1, kHeads, kHeadSize});
    auto k = q_param(ov::element::f16, ov::PartialShape{kBatch, -1, kHeads, kHeadSize});
    auto v = q_param(ov::element::f16, ov::PartialShape{kBatch, -1, kHeads, kHeadSize});
    auto cos = table_param();
    auto sin = table_param();
    auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, rotate_half_config());

    model = std::make_shared<ov::Model>(ov::OutputVector{plain_sdpa({rope, k, v})},
                                        ov::ParameterVector{q, k, v, cos, sin});
    manager.register_pass<RoPESDPAFusion>();
}

// IndirectSDPA reaches the primitive through a different creator and cannot take the table.
TEST_F(TransformationTestsF, RoPESDPAFusion_IndirectSDPANotFolded) {
    auto q = q_param();
    auto k = q_param();
    auto v = q_param();
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{kBatch});
    auto cos = table_param();
    auto sin = table_param();
    auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, rotate_half_config());
    auto sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(ov::OutputVector{rope, k, v},
                                                                 beam_idx,
                                                                 false,
                                                                 1,
                                                                 kIdentity,
                                                                 kIdentity,
                                                                 kIdentity,
                                                                 kIdentity,
                                                                 ov::element::dynamic);

    model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{q, k, v, beam_idx, cos, sin});
    manager.register_pass<RoPESDPAFusion>();
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
