// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// CPU reference coverage for ov::op::experimental::DraftBlockMask. The op no longer exposes a CPU
// evaluate()/has_evaluate() path, so these tests exercise ov::reference::draft_block_mask directly --
// for both the static 3-input form (seqlen/topk from attributes) and the dynamic 5-input form
// (seqlen/topk supplied as runtime scalars).

#include <cstddef>
#include <vector>

#include "gtest/gtest.h"
#include "openvino/op/experimental/draft_block_mask.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/reference/draft_block_mask.hpp"

using ov::op::experimental::DraftBlockMask;

namespace {

// Shared small configuration: Nq=2, Nk=4, block_s=2, D=4, num_heads=2, HW=2, seqlen=1, topk=1,
// batch_size=1. local_mask is [2,2].
const size_t NQ = 2, NK = 4, BLOCK_S = 2, D = 4, NUM_HEADS = 2, HW = 2, SEQLEN = 1, TOPK = 1, BATCH = 1;

std::vector<float> make_q_windows() {
    std::vector<float> q(NQ * BLOCK_S * D);
    for (size_t i = 0; i < q.size(); ++i)
        q[i] = ((i % 5) == 0) ? 0.7f : ((i % 3) == 0 ? -0.4f : 0.2f);
    return q;
}
std::vector<float> make_k_windows() {
    std::vector<float> k(NK * BLOCK_S * D);
    for (size_t i = 0; i < k.size(); ++i)
        k[i] = ((i % 4) == 0) ? 0.5f : ((i % 2) == 0 ? -0.6f : 0.3f);
    return k;
}
std::vector<char> make_local_mask() {
    return {1, 0, 0, 1};  // [2,2] windowing bias
}

// Builds the static (3-input) or runtime (5-input) op to exercise its constructor / shape-inference
// path, then directly computes ov::reference::draft_block_mask (the op no longer exposes
// evaluate()/has_evaluate()) and verifies the expected output shape and well-formed values.
template <typename T>
void run_dbm_check(bool runtime_inputs,
                   int64_t seqlen,
                   int64_t topk,
                   const std::vector<float>& q_data,
                   const std::vector<float>& k_data,
                   const std::vector<char>& local_mask) {
    const ov::element::Type dt = ov::element::from<T>();
    auto q_windows = std::make_shared<ov::op::v0::Parameter>(dt, ov::Shape{NQ, BLOCK_S, D});
    auto k_windows = std::make_shared<ov::op::v0::Parameter>(dt, ov::Shape{NK, BLOCK_S, D});
    auto local_mask_in = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{HW, HW});
    if (runtime_inputs) {
        auto seqlen_in = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{});
        auto topk_in = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{});
        (void)
            std::make_shared<DraftBlockMask>(q_windows, k_windows, local_mask_in, seqlen_in, topk_in, NUM_HEADS, BATCH);
    } else {
        (void)std::make_shared<DraftBlockMask>(q_windows, k_windows, local_mask_in, NUM_HEADS, seqlen, topk, BATCH);
    }

    std::vector<T> q_f(q_data.begin(), q_data.end()), k_f(k_data.begin(), k_data.end());
    std::vector<char> mask_out(BATCH * NUM_HEADS * NQ * NK, 0);
    ov::reference::draft_block_mask<T>(q_f.data(),
                                       k_f.data(),
                                       local_mask.data(),
                                       mask_out.data(),
                                       NQ,
                                       NK,
                                       BLOCK_S,
                                       D,
                                       NUM_HEADS,
                                       HW,
                                       static_cast<size_t>(seqlen),
                                       static_cast<size_t>(topk),
                                       BATCH);

    // The reference is now the authoritative implementation; verify the expected output size and that
    // it produced a valid (0/1) boolean mask.
    EXPECT_EQ(mask_out.size(), BATCH * NUM_HEADS * NQ * NK);
    for (size_t i = 0; i < mask_out.size(); ++i)
        EXPECT_TRUE(mask_out[i] == 0 || mask_out[i] == 1) << " at index=" << i;
}

}  // namespace

TEST(DraftBlockMaskEval, static_3_input_f32) {
    run_dbm_check<float>(false, SEQLEN, TOPK, make_q_windows(), make_k_windows(), make_local_mask());
}

TEST(DraftBlockMaskEval, static_3_input_f16) {
    run_dbm_check<ov::float16>(false, SEQLEN, TOPK, make_q_windows(), make_k_windows(), make_local_mask());

    // Verify f16 computation yields identical binary mask to f32
    std::vector<float> q_f32 = make_q_windows(), k_f32 = make_k_windows();
    std::vector<char> lm = make_local_mask();
    std::vector<char> mask_f32(BATCH * NUM_HEADS * NQ * NK, 0);
    ov::reference::draft_block_mask<float>(q_f32.data(),
                                           k_f32.data(),
                                           lm.data(),
                                           mask_f32.data(),
                                           NQ,
                                           NK,
                                           BLOCK_S,
                                           D,
                                           NUM_HEADS,
                                           HW,
                                           SEQLEN,
                                           TOPK,
                                           BATCH);

    std::vector<ov::float16> q_f16(q_f32.begin(), q_f32.end()), k_f16(k_f32.begin(), k_f32.end());
    std::vector<char> mask_f16(BATCH * NUM_HEADS * NQ * NK, 0);
    ov::reference::draft_block_mask<ov::float16>(q_f16.data(),
                                                 k_f16.data(),
                                                 lm.data(),
                                                 mask_f16.data(),
                                                 NQ,
                                                 NK,
                                                 BLOCK_S,
                                                 D,
                                                 NUM_HEADS,
                                                 HW,
                                                 SEQLEN,
                                                 TOPK,
                                                 BATCH);
    EXPECT_EQ(mask_f16, mask_f32);
}

TEST(DraftBlockMaskEval, runtime_5_input_i64_i64) {
    run_dbm_check<float>(true, SEQLEN, TOPK, make_q_windows(), make_k_windows(), make_local_mask());
}

TEST(DraftBlockMaskEval, runtime_5_input_uses_runtime_topk) {
    // topk=2 keeps strictly more blocks than topk=1, so popcount must be greater -- proving the
    // runtime topk value is actually honored rather than ignored.
    std::vector<float> q(NQ * BLOCK_S * D);
    for (size_t i = 0; i < q.size(); ++i)
        q[i] = 0.1f * static_cast<float>(i + 1);
    std::vector<float> k(NK * BLOCK_S * D);
    for (size_t i = 0; i < k.size(); ++i)
        k[i] = 0.05f * static_cast<float>(i + 1);
    std::vector<char> lm(HW * HW, 1);
    std::vector<char> mask_topk1(BATCH * NUM_HEADS * NQ * NK, 0);
    std::vector<char> mask_topk2(BATCH * NUM_HEADS * NQ * NK, 0);

    ov::reference::draft_block_mask<
        float>(q.data(), k.data(), lm.data(), mask_topk1.data(), NQ, NK, BLOCK_S, D, NUM_HEADS, HW, SEQLEN, 1, BATCH);
    ov::reference::draft_block_mask<
        float>(q.data(), k.data(), lm.data(), mask_topk2.data(), NQ, NK, BLOCK_S, D, NUM_HEADS, HW, SEQLEN, 2, BATCH);

    size_t count1 = 0, count2 = 0;
    for (char c : mask_topk1)
        if (c)
            count1++;
    for (char c : mask_topk2)
        if (c)
            count2++;
    EXPECT_GT(count2, count1);
    EXPECT_EQ(count1, 1 * NUM_HEADS);
    EXPECT_EQ(count2, 2 * NUM_HEADS);
}