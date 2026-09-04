// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/block_sparse_attention.hpp"

#include <gtest/gtest.h>

#include <random>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/reference/scaled_dot_product_attention.hpp"

using namespace reference_tests;
using namespace ov;

namespace {

// Deterministic (seeded) pseudo-random fill -- makes the very same test data reproducible
// across compilers/runs without committing large literal arrays to the source tree.
std::vector<float> makeData(size_t count, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(count);
    for (auto& v : data) {
        v = dist(gen);
    }
    return data;
}

// Runs the already-upstream, trusted ScaledDotProductAttention-13 reference kernel.
// BlockSparseAttention's own kernel must never be used to validate itself -- every expected
// value in this file is produced by this independent, pre-existing oracle instead.
std::vector<float> denseReference(const std::vector<float>& q,
                                   const std::vector<float>& k,
                                   const std::vector<float>& v,
                                   const Shape& q_shape,
                                   const Shape& k_shape,
                                   const Shape& v_shape,
                                   bool is_causal,
                                   const float* scale = nullptr) {
    Shape out_shape = q_shape;
    out_shape.back() = v_shape.back();
    std::vector<float> out(shape_size(out_shape), 0.0f);
    ov::reference::scaled_dot_product_attention<float, char>(q.data(),
                                                              k.data(),
                                                              v.data(),
                                                              nullptr,
                                                              scale,
                                                              nullptr,
                                                              out.data(),
                                                              is_causal,
                                                              q_shape,
                                                              k_shape,
                                                              v_shape,
                                                              Shape{},
                                                              Shape{},
                                                              out_shape);
    return out;
}

// Concatenates the requested blocks (each `block_size` long along the sequence axis) out of a
// [B, Hk, S, E] tensor into a shorter, contiguous [B, Hk, blocks.size()*block_size, E] tensor.
// This is the "obviously correct" (if inefficient) mathematical definition of block-sparse
// attention with a *uniform* selection (same blocks picked for every query position): gather the
// selected blocks, then run plain dense attention over just those tokens. It is intentionally a
// different code path from BlockSparseAttention's own direct-indexed reference kernel.
std::vector<float> gatherBlocks(const std::vector<float>& tensor,
                                 const Shape& shape,
                                 const std::vector<int64_t>& blocks,
                                 int64_t block_size) {
    const auto B = static_cast<int64_t>(shape[0]);
    const auto Hk = static_cast<int64_t>(shape[1]);
    const auto S = static_cast<int64_t>(shape[2]);
    const auto E = static_cast<int64_t>(shape[3]);
    std::vector<float> out(static_cast<size_t>(B * Hk * static_cast<int64_t>(blocks.size()) * block_size * E));
    size_t o = 0;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < Hk; ++h) {
            for (const auto blk : blocks) {
                const size_t base = static_cast<size_t>(((b * Hk + h) * S + blk * block_size) * E);
                const size_t len = static_cast<size_t>(block_size * E);
                std::copy(tensor.begin() + static_cast<std::ptrdiff_t>(base),
                          tensor.begin() + static_cast<std::ptrdiff_t>(base + len),
                          out.begin() + static_cast<std::ptrdiff_t>(o));
                o += len;
            }
        }
    }
    return out;
}

// Independent oracle for the case where DIFFERENT query blocks select DIFFERENT sets of key
// blocks (the realistic case for e.g. a "local window + global sink block" sparsity pattern, as
// used by FlashVSR-style long-context video attention). For each query block this gathers just
// that row-group's selected blocks and calls the trusted dense SDPA reference on the resulting
// short sequence -- BlockSparseAttention's own kernel is not involved in producing this value.
std::vector<float> referenceViaPerBlockGather(const std::vector<float>& q,
                                               const std::vector<float>& k,
                                               const std::vector<float>& v,
                                               const Shape& q_shape,
                                               const Shape& k_shape,
                                               const Shape& v_shape,
                                               const std::vector<std::vector<int64_t>>& blocksPerQueryBlock,
                                               int64_t block_size) {
    const auto B = static_cast<int64_t>(q_shape[0]);
    const auto H = static_cast<int64_t>(q_shape[1]);
    const auto L = static_cast<int64_t>(q_shape[2]);
    const auto E = static_cast<int64_t>(q_shape[3]);
    const auto Hk = static_cast<int64_t>(k_shape[1]);
    const auto S = static_cast<int64_t>(k_shape[2]);
    const auto Ev = static_cast<int64_t>(v_shape[3]);
    const auto num_q_blocks = L / block_size;

    std::vector<float> out(static_cast<size_t>(B * H * L * Ev), 0.0f);

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            const int64_t hk = (Hk == 1) ? 0 : h;
            for (int64_t qb = 0; qb < num_q_blocks; ++qb) {
                const auto& blocks = blocksPerQueryBlock[static_cast<size_t>(qb)];

                std::vector<float> k_gathered(static_cast<size_t>(static_cast<int64_t>(blocks.size()) * block_size * E));
                std::vector<float> v_gathered(
                    static_cast<size_t>(static_cast<int64_t>(blocks.size()) * block_size * Ev));
                for (size_t bi = 0; bi < blocks.size(); ++bi) {
                    const int64_t blk = blocks[bi];
                    const float* k_src = k.data() + static_cast<size_t>(((b * Hk + hk) * S + blk * block_size) * E);
                    const float* v_src = v.data() + static_cast<size_t>(((b * Hk + hk) * S + blk * block_size) * Ev);
                    std::copy(k_src,
                              k_src + static_cast<size_t>(block_size * E),
                              k_gathered.begin() + static_cast<std::ptrdiff_t>(bi * static_cast<size_t>(block_size * E)));
                    std::copy(
                        v_src,
                        v_src + static_cast<size_t>(block_size * Ev),
                        v_gathered.begin() + static_cast<std::ptrdiff_t>(bi * static_cast<size_t>(block_size * Ev)));
                }

                std::vector<float> q_block(static_cast<size_t>(block_size * E));
                const float* q_src = q.data() + static_cast<size_t>(((b * H + h) * L + qb * block_size) * E);
                std::copy(q_src, q_src + static_cast<size_t>(block_size * E), q_block.begin());

                const Shape qb_shape{1, 1, static_cast<size_t>(block_size), static_cast<size_t>(E)};
                const Shape kb_shape{1, 1, blocks.size() * static_cast<size_t>(block_size), static_cast<size_t>(E)};
                const Shape vb_shape{1, 1, blocks.size() * static_cast<size_t>(block_size), static_cast<size_t>(Ev)};
                const Shape ob_shape{1, 1, static_cast<size_t>(block_size), static_cast<size_t>(Ev)};

                std::vector<float> ob(shape_size(ob_shape), 0.0f);
                ov::reference::scaled_dot_product_attention<float, char>(q_block.data(),
                                                                          k_gathered.data(),
                                                                          v_gathered.data(),
                                                                          nullptr,
                                                                          nullptr,
                                                                          nullptr,
                                                                          ob.data(),
                                                                          false,
                                                                          qb_shape,
                                                                          kb_shape,
                                                                          vb_shape,
                                                                          Shape{},
                                                                          Shape{},
                                                                          ob_shape);

                float* out_dst = out.data() + static_cast<size_t>(((b * H + h) * L + qb * block_size) * Ev);
                std::copy(ob.begin(), ob.end(), out_dst);
            }
        }
    }
    return out;
}

// Replicates a single shared kv-head `Hk == 1` tensor into `H` full copies -- used to build an
// oracle input for the head-broadcast test without relying on any broadcasting behaviour of the
// dense reference kernel itself.
std::vector<float> expandSharedHead(const std::vector<float>& tensor, const Shape& shape, int64_t H) {
    const auto B = static_cast<int64_t>(shape[0]);
    const auto S = static_cast<int64_t>(shape[2]);
    const auto E = static_cast<int64_t>(shape[3]);
    std::vector<float> out(static_cast<size_t>(B * H * S * E));
    for (int64_t b = 0; b < B; ++b) {
        const float* src = tensor.data() + static_cast<size_t>(b * S * E);
        for (int64_t h = 0; h < H; ++h) {
            float* dst = out.data() + static_cast<size_t>((b * H + h) * S * E);
            std::copy(src, src + static_cast<size_t>(S * E), dst);
        }
    }
    return out;
}

}  // namespace

class ReferenceBlockSparseAttentionTest : public testing::Test, public CommonReferenceTest {
protected:
    void RunTest(const Shape& qShape,
                 const std::vector<float>& qData,
                 const Shape& kShape,
                 const std::vector<float>& kData,
                 const Shape& vShape,
                 const std::vector<float>& vData,
                 const Shape& biShape,
                 const std::vector<int64_t>& biData,
                 bool causal,
                 int64_t blockSize,
                 const Shape& maskShape,
                 const std::vector<char>& maskData,
                 const float* scaleValue,
                 const std::vector<float>& expected) {
        const auto q = std::make_shared<op::v0::Parameter>(element::f32, qShape);
        const auto k = std::make_shared<op::v0::Parameter>(element::f32, kShape);
        const auto v = std::make_shared<op::v0::Parameter>(element::f32, vShape);
        const auto bi = std::make_shared<op::v0::Parameter>(element::i64, biShape);

        OutputVector inputs = {q, k, v, bi};
        ParameterVector params = {q, k, v, bi};
        inputData = {CreateTensor(qShape, element::f32, qData),
                     CreateTensor(kShape, element::f32, kData),
                     CreateTensor(vShape, element::f32, vData),
                     CreateTensor(biShape, element::i64, biData)};

        if (!maskShape.empty()) {
            const auto mask = std::make_shared<op::v0::Parameter>(element::boolean, maskShape);
            inputs.push_back(mask);
            params.push_back(mask);
            inputData.push_back(CreateTensor(maskShape, element::boolean, maskData));
        } else if (scaleValue) {
            // The op resolves optional trailing inputs purely by count: 5 inputs always means
            // "has mask, no scale" and 6 always means "has mask and scale" -- there is no
            // positional form for "scale but no mask". So whenever a scale is requested without
            // an explicit mask, synthesize a trivial all-true mask to keep the input list valid.
            const std::vector<char> allTrueMask(shape_size(biShape), 1);
            const auto mask = std::make_shared<op::v0::Parameter>(element::boolean, biShape);
            inputs.push_back(mask);
            params.push_back(mask);
            inputData.push_back(CreateTensor(biShape, element::boolean, allTrueMask));
        }

        if (scaleValue) {
            const auto scale = std::make_shared<op::v0::Constant>(element::f32, Shape{}, *scaleValue);
            inputs.push_back(scale);
        }

        const auto op = std::make_shared<ov::op::v17::BlockSparseAttention>(inputs, blockSize, causal);
        function = std::make_shared<Model>(OutputVector{op}, params);

        refOutData = {CreateTensor(qShape, element::f32, expected)};
        Exec();
    }
};

// All blocks selected for every query position + causal=false must be bit-for-bit equivalent to
// plain dense attention: the strongest possible cross-check against the trusted SDPA reference,
// since there is no sparsity left to reason about independently.
TEST_F(ReferenceBlockSparseAttentionTest, DenseEquivalent_NonCausal) {
    constexpr int64_t B = 1, H = 2, L = 8, E = 4, Ev = 4, blockSize = 2, numKvBlocks = 4;
    const Shape qShape{B, H, L, E}, kShape{B, H, L, E}, vShape{B, H, L, Ev};
    const auto qData = makeData(static_cast<size_t>(B * H * L * E), 1);
    const auto kData = makeData(static_cast<size_t>(B * H * L * E), 2);
    const auto vData = makeData(static_cast<size_t>(B * H * L * Ev), 3);

    const int64_t numQBlocks = L / blockSize;
    const Shape biShape{B, H, static_cast<size_t>(numQBlocks), numKvBlocks};
    std::vector<int64_t> biData;
    for (int64_t i = 0; i < B * H * numQBlocks; ++i) {
        for (int64_t blk = 0; blk < numKvBlocks; ++blk) {
            biData.push_back(blk);
        }
    }

    const auto expected = denseReference(qData, kData, vData, qShape, kShape, vShape, /*is_causal=*/false);

    RunTest(qShape, qData, kShape, kData, vShape, vData, biShape, biData, /*causal=*/false, blockSize, {}, {}, nullptr, expected);
}

// Same as above but with the op's own `causal` attribute set. Every query block still lists
// *all* kv blocks (including "future" ones) -- the reference kernel's intra-block causal
// truncation must make this reduce exactly to standard causal dense attention.
TEST_F(ReferenceBlockSparseAttentionTest, DenseEquivalent_Causal) {
    constexpr int64_t B = 1, H = 2, L = 8, E = 4, Ev = 4, blockSize = 2, numKvBlocks = 4;
    const Shape qShape{B, H, L, E}, kShape{B, H, L, E}, vShape{B, H, L, Ev};
    const auto qData = makeData(static_cast<size_t>(B * H * L * E), 11);
    const auto kData = makeData(static_cast<size_t>(B * H * L * E), 12);
    const auto vData = makeData(static_cast<size_t>(B * H * L * Ev), 13);

    const int64_t numQBlocks = L / blockSize;
    const Shape biShape{B, H, static_cast<size_t>(numQBlocks), numKvBlocks};
    std::vector<int64_t> biData;
    for (int64_t i = 0; i < B * H * numQBlocks; ++i) {
        for (int64_t blk = 0; blk < numKvBlocks; ++blk) {
            biData.push_back(blk);
        }
    }

    const auto expected = denseReference(qData, kData, vData, qShape, kShape, vShape, /*is_causal=*/true);

    RunTest(qShape, qData, kShape, kData, vShape, vData, biShape, biData, /*causal=*/true, blockSize, {}, {}, nullptr, expected);
}

// Every query block selects the SAME fixed, non-contiguous pair of kv blocks {0, 3} out of 4:
// the canonical block-sparse scenario. Expected output is computed by gathering just those two
// blocks and running dense attention on the resulting short sequence.
TEST_F(ReferenceBlockSparseAttentionTest, SparseUniformSelection_NonCausal) {
    constexpr int64_t B = 1, H = 2, L = 8, E = 4, Ev = 4, blockSize = 2;
    const Shape qShape{B, H, L, E}, kShape{B, H, L, E}, vShape{B, H, L, Ev};
    const auto qData = makeData(static_cast<size_t>(B * H * L * E), 21);
    const auto kData = makeData(static_cast<size_t>(B * H * L * E), 22);
    const auto vData = makeData(static_cast<size_t>(B * H * L * Ev), 23);

    const std::vector<int64_t> selected{0, 3};
    const int64_t numQBlocks = L / blockSize;
    const Shape biShape{B, H, static_cast<size_t>(numQBlocks), selected.size()};
    std::vector<int64_t> biData;
    for (int64_t i = 0; i < B * H * numQBlocks; ++i) {
        biData.insert(biData.end(), selected.begin(), selected.end());
    }

    const auto kGathered = gatherBlocks(kData, kShape, selected, blockSize);
    const auto vGathered = gatherBlocks(vData, vShape, selected, blockSize);
    const Shape kGatheredShape{B, H, selected.size() * blockSize, E};
    const Shape vGatheredShape{B, H, selected.size() * blockSize, Ev};
    const auto expected =
        denseReference(qData, kGathered, vGathered, qShape, kGatheredShape, vGatheredShape, /*is_causal=*/false);

    RunTest(qShape, qData, kShape, kData, vShape, vData, biShape, biData, /*causal=*/false, blockSize, {}, {}, nullptr, expected);
}

// Adds a third, masked-off candidate slot per row that duplicates an already-selected block
// index. If the mask were ignored, that block's contribution would be counted twice and the
// softmax weights would visibly diverge from the 2-block gathered oracle used here.
TEST_F(ReferenceBlockSparseAttentionTest, PaddingMaskIgnoresDuplicateSlot) {
    constexpr int64_t B = 1, H = 2, L = 8, E = 4, Ev = 4, blockSize = 2;
    const Shape qShape{B, H, L, E}, kShape{B, H, L, E}, vShape{B, H, L, Ev};
    const auto qData = makeData(static_cast<size_t>(B * H * L * E), 31);
    const auto kData = makeData(static_cast<size_t>(B * H * L * E), 32);
    const auto vData = makeData(static_cast<size_t>(B * H * L * Ev), 33);

    const std::vector<int64_t> selected{0, 3};
    const int64_t numQBlocks = L / blockSize;
    const int64_t kBlocks = 3;
    const Shape biShape{B, H, static_cast<size_t>(numQBlocks), static_cast<size_t>(kBlocks)};
    std::vector<int64_t> biData;
    std::vector<char> maskData;
    for (int64_t i = 0; i < B * H * numQBlocks; ++i) {
        biData.insert(biData.end(), {0, 3, 0});   // 3rd slot duplicates block 0
        maskData.insert(maskData.end(), {1, 1, 0});  // ... but is masked off
    }

    const auto kGathered = gatherBlocks(kData, kShape, selected, blockSize);
    const auto vGathered = gatherBlocks(vData, vShape, selected, blockSize);
    const Shape kGatheredShape{B, H, selected.size() * blockSize, E};
    const Shape vGatheredShape{B, H, selected.size() * blockSize, Ev};
    const auto expected =
        denseReference(qData, kGathered, vGathered, qShape, kGatheredShape, vGatheredShape, /*is_causal=*/false);

    RunTest(qShape,
            qData,
            kShape,
            kData,
            vShape,
            vData,
            biShape,
            biData,
            /*causal=*/false,
            blockSize,
            biShape,
            maskData,
            nullptr,
            expected);
}

// Explicit scale attribute must actually be used by the kernel, not silently replaced by the
// default `1/sqrt(E)` -- E=4 makes the default exactly 0.5, so 0.25 is unambiguous evidence the
// override took effect.
TEST_F(ReferenceBlockSparseAttentionTest, ExplicitScaleOverride) {
    constexpr int64_t B = 1, H = 2, L = 8, E = 4, Ev = 4, blockSize = 2, numKvBlocks = 4;
    const Shape qShape{B, H, L, E}, kShape{B, H, L, E}, vShape{B, H, L, Ev};
    const auto qData = makeData(static_cast<size_t>(B * H * L * E), 41);
    const auto kData = makeData(static_cast<size_t>(B * H * L * E), 42);
    const auto vData = makeData(static_cast<size_t>(B * H * L * Ev), 43);

    const int64_t numQBlocks = L / blockSize;
    const Shape biShape{B, H, static_cast<size_t>(numQBlocks), numKvBlocks};
    std::vector<int64_t> biData;
    for (int64_t i = 0; i < B * H * numQBlocks; ++i) {
        for (int64_t blk = 0; blk < numKvBlocks; ++blk) {
            biData.push_back(blk);
        }
    }

    const float scaleValue = 0.25f;
    const auto expected =
        denseReference(qData, kData, vData, qShape, kShape, vShape, /*is_causal=*/false, &scaleValue);

    RunTest(qShape,
            qData,
            kShape,
            kData,
            vShape,
            vData,
            biShape,
            biData,
            /*causal=*/false,
            blockSize,
            {},
            {},
            &scaleValue,
            expected);
}

// H=4 query heads sharing a single (Hk=1) kv head and a single (Hb=1) block-indices head --
// SDPA's own broadcast contract, reused here. Expected output is built by first manually
// expanding the shared kv head into 4 explicit copies (a dead-simple copy loop, independent of
// any broadcasting logic in either kernel) and then calling the dense oracle.
TEST_F(ReferenceBlockSparseAttentionTest, HeadBroadcast_SharedKvAndBlockIndices) {
    constexpr int64_t B = 1, H = 4, Hk = 1, L = 8, S = 8, E = 4, Ev = 4, blockSize = 2, numKvBlocks = 4;
    const Shape qShape{B, H, L, E};
    const Shape kShape{B, Hk, S, E};
    const Shape vShape{B, Hk, S, Ev};
    const auto qData = makeData(static_cast<size_t>(B * H * L * E), 51);
    const auto kData = makeData(static_cast<size_t>(B * Hk * S * E), 52);
    const auto vData = makeData(static_cast<size_t>(B * Hk * S * Ev), 53);

    const int64_t numQBlocks = L / blockSize;
    const Shape biShape{B, 1, static_cast<size_t>(numQBlocks), numKvBlocks};  // Hb=1: shared across heads
    std::vector<int64_t> biData;
    for (int64_t i = 0; i < B * 1 * numQBlocks; ++i) {
        for (int64_t blk = 0; blk < numKvBlocks; ++blk) {
            biData.push_back(blk);
        }
    }

    const auto kExpanded = expandSharedHead(kData, kShape, H);
    const auto vExpanded = expandSharedHead(vData, vShape, H);
    const Shape kExpandedShape{B, H, S, E};
    const Shape vExpandedShape{B, H, S, Ev};
    const auto expected =
        denseReference(qData, kExpanded, vExpanded, qShape, kExpandedShape, vExpandedShape, /*is_causal=*/false);

    RunTest(qShape, qData, kShape, kData, vShape, vData, biShape, biData, /*causal=*/false, blockSize, {}, {}, nullptr, expected);
}

// Realistic "local window + global sink block" sparsity pattern, structurally representative of
// how long-context video models such as FlashVSR restrict attention to a handful of local and
// global blocks per query position instead of the full sequence. Every query block selects a
// *different* set of kv blocks (own block, previous block, and the first "global" block, with
// duplicates masked off) -- exercised via the per-block gather oracle since a single dense call
// cannot express varying selections.
TEST_F(ReferenceBlockSparseAttentionTest, LocalWindowPlusGlobalSink_VaryingSelection) {
    constexpr int64_t B = 1, H = 1, L = 16, S = 16, E = 4, Ev = 4, blockSize = 2;
    const int64_t numQBlocks = L / blockSize;
    const int64_t numKvBlocks = S / blockSize;
    const Shape qShape{B, H, L, E}, kShape{B, H, S, E}, vShape{B, H, S, Ev};
    const auto qData = makeData(static_cast<size_t>(B * H * L * E), 61);
    const auto kData = makeData(static_cast<size_t>(B * H * S * E), 62);
    const auto vData = makeData(static_cast<size_t>(B * H * S * Ev), 63);

    constexpr int64_t kBlocks = 3;  // slots: [global=0, prev=qb-1, own=qb]
    const Shape biShape{B, H, static_cast<size_t>(numQBlocks), static_cast<size_t>(kBlocks)};
    std::vector<int64_t> biData;
    std::vector<char> maskData;
    std::vector<std::vector<int64_t>> blocksPerQueryBlock(static_cast<size_t>(numQBlocks));

    for (int64_t qb = 0; qb < numQBlocks; ++qb) {
        const int64_t own = qb;
        const int64_t prev = std::max<int64_t>(qb - 1, 0);
        const int64_t global = 0;
        const char ownMask = 1;
        const char prevMask = (qb >= 1) ? 1 : 0;                         // no previous block at qb==0
        const char globalMask = (global != prev && global != own) ? 1 : 0;  // avoid double counting

        biData.insert(biData.end(), {global, prev, own});
        maskData.insert(maskData.end(), {globalMask, prevMask, ownMask});

        std::vector<int64_t> active;
        if (globalMask) {
            active.push_back(global);
        }
        if (prevMask) {
            active.push_back(prev);
        }
        active.push_back(own);
        blocksPerQueryBlock[static_cast<size_t>(qb)] = active;
    }
    ASSERT_EQ(static_cast<int64_t>(numKvBlocks), numQBlocks);  // sanity: square attention in this test

    const auto expected =
        referenceViaPerBlockGather(qData, kData, vData, qShape, kShape, vShape, blocksPerQueryBlock, blockSize);

    RunTest(qShape,
            qData,
            kShape,
            kData,
            vShape,
            vData,
            biShape,
            biData,
            /*causal=*/false,
            blockSize,
            biShape,
            maskData,
            nullptr,
            expected);
}

// Exercises the i32 block_indices dispatch path (all other cases above use i64) -- a smaller
// repeat of the dense-equivalent scenario is enough since the numerical kernel code is identical
// regardless of index width.
TEST_F(ReferenceBlockSparseAttentionTest, Int32BlockIndices) {
    constexpr int64_t B = 1, H = 1, L = 4, E = 4, Ev = 4, blockSize = 2, numKvBlocks = 2;
    const Shape qShape{B, H, L, E}, kShape{B, H, L, E}, vShape{B, H, L, Ev};
    const auto qData = makeData(static_cast<size_t>(B * H * L * E), 71);
    const auto kData = makeData(static_cast<size_t>(B * H * L * E), 72);
    const auto vData = makeData(static_cast<size_t>(B * H * L * Ev), 73);

    const int64_t numQBlocks = L / blockSize;
    const Shape biShape{B, H, static_cast<size_t>(numQBlocks), numKvBlocks};

    const auto q = std::make_shared<op::v0::Parameter>(element::f32, qShape);
    const auto k = std::make_shared<op::v0::Parameter>(element::f32, kShape);
    const auto v = std::make_shared<op::v0::Parameter>(element::f32, vShape);
    const auto bi = std::make_shared<op::v0::Parameter>(element::i32, biShape);

    std::vector<int32_t> biData;
    for (int64_t i = 0; i < B * H * numQBlocks; ++i) {
        for (int64_t blk = 0; blk < numKvBlocks; ++blk) {
            biData.push_back(static_cast<int32_t>(blk));
        }
    }
    const auto expected = denseReference(qData, kData, vData, qShape, kShape, vShape, /*is_causal=*/false);

    const auto op =
        std::make_shared<ov::op::v17::BlockSparseAttention>(OutputVector{q, k, v, bi}, blockSize, /*causal=*/false);
    function = std::make_shared<Model>(OutputVector{op}, ParameterVector{q, k, v, bi});
    inputData = {CreateTensor(qShape, element::f32, qData),
                 CreateTensor(kShape, element::f32, kData),
                 CreateTensor(vShape, element::f32, vData),
                 CreateTensor(biShape, element::i32, biData)};
    refOutData = {CreateTensor(qShape, element::f32, expected)};
    Exec();
}
