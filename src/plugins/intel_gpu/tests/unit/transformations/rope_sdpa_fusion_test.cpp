// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "intel_gpu/op/indirect_sdpa.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
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
// [batch, tokens, heads, head_size] -> BHLS. The fused rotation reads the table with the
// sequence length and head size this order selects, so it is the only one the pass accepts.
const std::vector<int64_t> kBtnsToBhls{0, 2, 1, 3};

ov::op::internal::RoPE::Config interleaved_config() {
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

std::shared_ptr<ov::intel_gpu::op::SDPA> plain_sdpa(const ov::OutputVector& inputs,
                                                    const std::vector<int64_t>& order_q = kBtnsToBhls) {
    return std::make_shared<ov::intel_gpu::op::SDPA>(inputs,
                                                     false,
                                                     order_q,
                                                     kIdentity,
                                                     kIdentity,
                                                     kIdentity,
                                                     ov::element::dynamic);
}

std::shared_ptr<ov::intel_gpu::op::SDPA> rope_sdpa(const ov::OutputVector& inputs) {
    return std::make_shared<ov::intel_gpu::op::SDPA>(inputs,
                                                     false,
                                                     kBtnsToBhls,
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
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, interleaved_config());

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
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, interleaved_config());

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
    auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{k, cos, sin}, interleaved_config());

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
    auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, interleaved_config());

    model = std::make_shared<ov::Model>(ov::OutputVector{plain_sdpa({rope, k, v}), rope},
                                        ov::ParameterVector{q, k, v, cos, sin});
    manager.register_pass<RoPESDPAFusion>();
}

// The fused rotation tile-loads the tables as f16, so an f32 table alone is enough to decline.
TEST_F(TransformationTestsF, RoPESDPAFusion_F32TablesNotFolded) {
    auto q = q_param();
    auto k = q_param();
    auto v = q_param();
    auto cos = table_param(ov::element::f32);
    auto sin = table_param(ov::element::f32);
    auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, interleaved_config());

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
    auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, interleaved_config());

    model = std::make_shared<ov::Model>(ov::OutputVector{plain_sdpa({rope, k, v})},
                                        ov::ParameterVector{q, k, v, cos, sin});
    manager.register_pass<RoPESDPAFusion>();
}

// IndirectSDPA reaches the primitive through a different creator and cannot take the table.
// Not a TransformationTestsF case: that fixture clones the model to build its reference, and
// IndirectSDPA::clone_with_new_inputs swaps its compressed and uncompressed branches, so cloning
// an uncompressed one throws for reasons that have nothing to do with this pass.
TEST(RoPESDPAFusionTest, IndirectSDPANotFolded) {
    auto q = q_param();
    auto k = q_param();
    auto v = q_param();
    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{kBatch});
    auto cos = table_param();
    auto sin = table_param();
    auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, interleaved_config());
    auto sdpa = std::make_shared<ov::intel_gpu::op::IndirectSDPA>(ov::OutputVector{rope, k, v},
                                                                 beam_idx,
                                                                 false,
                                                                 1,
                                                                 kBtnsToBhls,
                                                                 kIdentity,
                                                                 kIdentity,
                                                                 kIdentity,
                                                                 ov::element::dynamic);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{sdpa},
                                             ov::ParameterVector{q, k, v, beam_idx, cos, sin});

    ov::pass::Manager manager;
    manager.register_pass<RoPESDPAFusion>();
    manager.run_passes(model);

    EXPECT_EQ(count_ops_of_type<ov::op::internal::RoPE>(model), 1u);
    auto kept = ov::as_type_ptr<ov::intel_gpu::op::IndirectSDPA>(
        model->get_results()[0]->get_input_node_shared_ptr(0));
    ASSERT_TRUE(kept);
    EXPECT_EQ(kept->get_input_size(), 4u);
    EXPECT_FALSE(kept->get_rope_q());
}

// The kernel reads the cos/sin table with the sequence length and head size it reaches through
// input_q_transpose_order. Under any other order those are different axes of Q than the ones the
// pattern measured the table against, so the kernel would index the table along the wrong axis --
// here it would read kHeads rows of a kTokens-row table.
TEST_F(TransformationTestsF, RoPESDPAFusion_NonCanonicalTransposeOrderNotFolded) {
    auto q = q_param();
    auto k = q_param();
    auto v = q_param();
    auto cos = table_param();
    auto sin = table_param();
    auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, interleaved_config());

    model = std::make_shared<ov::Model>(ov::OutputVector{plain_sdpa({rope, k, v}, kIdentity)},
                                        ov::ParameterVector{q, k, v, cos, sin});
    manager.register_pass<RoPESDPAFusion>();
}

// An i8 key selects the integer K^T*Q contraction, which needs Q to hold exact integer codes.
// A rotation mixes two codes per element, so the rounding that packs Q into SLM would saturate.
TEST_F(TransformationTestsF, RoPESDPAFusion_I8KeyNotFolded) {
    auto q = q_param();
    auto k = q_param(ov::element::i8);
    auto v = q_param();
    auto cos = table_param();
    auto sin = table_param();
    auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, interleaved_config());

    model = std::make_shared<ov::Model>(ov::OutputVector{plain_sdpa({rope, k, v})},
                                        ov::ParameterVector{q, k, v, cos, sin});
    manager.register_pass<RoPESDPAFusion>();
}

// The kernel implements the interleaved rotation on (2i, 2i+1) pairs only.
TEST_F(TransformationTestsF, RoPESDPAFusion_RotateHalfNotFolded) {
    auto config = interleaved_config();
    config.is_interleaved = false;

    auto q = q_param();
    auto k = q_param();
    auto v = q_param();
    auto cos = table_param();
    auto sin = table_param();
    auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, config);

    model = std::make_shared<ov::Model>(ov::OutputVector{plain_sdpa({rope, k, v})},
                                        ov::ParameterVector{q, k, v, cos, sin});
    manager.register_pass<RoPESDPAFusion>();
}

// The fused path rotates the whole head, so a partial rotary width has an unrotated tail it
// would have to carry through untouched.
TEST_F(TransformationTestsF, RoPESDPAFusion_PartialRotaryNotFolded) {
    auto config = interleaved_config();
    config.rotary_ndims = kHeadSize / 2;

    auto q = q_param();
    auto k = q_param();
    auto v = q_param();
    auto cos = table_param();
    auto sin = table_param();
    auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, config);

    model = std::make_shared<ov::Model>(ov::OutputVector{plain_sdpa({rope, k, v})},
                                        ov::ParameterVector{q, k, v, cos, sin});
    manager.register_pass<RoPESDPAFusion>();
}

// A gathered table is addressed through a position index the fused rotation does not read.
TEST_F(TransformationTestsF, RoPESDPAFusion_GatheredPositionsNotFolded) {
    auto config = interleaved_config();
    config.gather_position_arg_id = 3;

    auto q = q_param();
    auto k = q_param();
    auto v = q_param();
    auto cos = table_param();
    auto sin = table_param();
    auto positions = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{kBatch, kTokens});
    auto rope =
        std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin, positions}, config);

    model = std::make_shared<ov::Model>(ov::OutputVector{plain_sdpa({rope, k, v})},
                                        ov::ParameterVector{q, k, v, cos, sin, positions});
    manager.register_pass<RoPESDPAFusion>();
}

// A sliced input means the RoPE reads a window of a larger tensor, which the fused rotation,
// reading Q straight from the SDPA input, cannot reproduce.
TEST_F(TransformationTestsF, RoPESDPAFusion_SlicedInputNotFolded) {
    auto config = interleaved_config();
    config.slice_start = 0;
    config.slice_stop = kHeadSize;

    auto q = q_param();
    auto k = q_param();
    auto v = q_param();
    auto cos = table_param();
    auto sin = table_param();
    auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, config);

    model = std::make_shared<ov::Model>(ov::OutputVector{plain_sdpa({rope, k, v})},
                                        ov::ParameterVector{q, k, v, cos, sin});
    manager.register_pass<RoPESDPAFusion>();
}

// The kernel reads the table densely as (batch, tokens, head_size). A table holding the same
// number of elements in a different arrangement is addressed differently by the reference RoPE
// kernel and by the fused one, so an element-count match is not enough to accept it.
TEST_F(TransformationTestsF, RoPESDPAFusion_RepackedTableNotFolded) {
    auto q = q_param();
    auto k = q_param();
    auto v = q_param();
    auto cos = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f16, ov::PartialShape{kBatch, kTokens * kHeads, kHeadSize / kHeads});
    auto sin = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f16, ov::PartialShape{kBatch, kTokens * kHeads, kHeadSize / kHeads});
    auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, interleaved_config());

    model = std::make_shared<ov::Model>(ov::OutputVector{plain_sdpa({rope, k, v})},
                                        ov::ParameterVector{q, k, v, cos, sin});
    manager.register_pass<RoPESDPAFusion>();
}

// A Convert between the rotation and attention breaks the match. Upstream's f32 pinning of
// rotary sine/cosine chains puts one there, so this is the shape that silently turns the pass
// into a no-op rather than a hypothetical.
TEST_F(TransformationTestsF, RoPESDPAFusion_ConvertBetweenRoPEAndSDPANotFolded) {
    auto q = q_param();
    auto k = q_param();
    auto v = q_param();
    auto cos = table_param();
    auto sin = table_param();
    auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, interleaved_config());
    auto convert = std::make_shared<ov::op::v0::Convert>(rope, ov::element::f16);

    model = std::make_shared<ov::Model>(ov::OutputVector{plain_sdpa({convert, k, v})},
                                        ov::ParameterVector{q, k, v, cos, sin});
    manager.register_pass<RoPESDPAFusion>();
}

// An SDPA that already carries a fused rotation must not be handed a second pair of tables:
// the generator reads cos/sin as the last two inputs, so a second pair would displace them.
TEST_F(TransformationTestsF, RoPESDPAFusion_AlreadyFusedNotFolded) {
    auto q = q_param();
    auto k = q_param();
    auto v = q_param();
    auto cos = table_param();
    auto sin = table_param();
    auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, interleaved_config());

    model = std::make_shared<ov::Model>(ov::OutputVector{rope_sdpa({rope, k, v, cos, sin})},
                                        ov::ParameterVector{q, k, v, cos, sin});
    manager.register_pass<RoPESDPAFusion>();
}

// Same element count, different arrangement: with one token per batch a [1, batch, head_size]
// table has exactly as many elements as the [batch, 1, head_size] one the kernel expects, and
// matching the shape by value rather than by position accepted it. The kernel strides the table
// by head_size per token and by tokens*head_size per batch, so it would read batch 1 out of the
// row belonging to token 1.
TEST_F(TransformationTestsF, RoPESDPAFusion_SwappedTableAxesNotFolded) {
    constexpr int64_t kTwoBatches = 2;
    const ov::PartialShape q_shape{kTwoBatches, 1, kHeads, kHeadSize};

    auto config = interleaved_config();
    auto q = q_param(ov::element::f16, q_shape);
    auto k = q_param(ov::element::f16, q_shape);
    auto v = q_param(ov::element::f16, q_shape);
    auto cos = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                       ov::PartialShape{1, kTwoBatches, kHeadSize});
    auto sin = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                       ov::PartialShape{1, kTwoBatches, kHeadSize});
    auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{q, cos, sin}, config);

    model = std::make_shared<ov::Model>(ov::OutputVector{plain_sdpa({rope, k, v})},
                                        ov::ParameterVector{q, k, v, cos, sin});
    manager.register_pass<RoPESDPAFusion>();
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
