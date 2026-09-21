// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe/moe_infer_utils.hpp"

#include <gtest/gtest.h>

#include "openvino/runtime/make_tensor.hpp"

/*
 * Regression tests for the accumulator-clearing optimization introduced to replace an
 * unconditional memset of the whole MoE output accumulator (see run_expert_iterative()):
 * only cells that scatter_expert_outputs() did not write (because a token's real routing
 * count fell short of num_active_experts, e.g. due to is_nonzero() filtering) get cleared
 * afterwards. These tests pin the exact behavior clear_unfilled_accumulator_slots() must
 * preserve: filled cells are left untouched, unfilled cells are zeroed, regardless of
 * whatever garbage previously occupied the buffer.
 */

namespace {

using ov::npuw::moe::clear_unfilled_accumulator_slots;
using ov::npuw::moe::scatter_expert_outputs;

constexpr size_t kNumActiveExperts = 3;  // K
constexpr size_t kNumTokens = 4;
constexpr size_t kEmbedDim = 2;

ov::SoPtr<ov::ITensor> make_accumulator(float fill_value) {
    ov::Tensor t(ov::element::f32, ov::Shape{kNumActiveExperts, 1, kNumTokens, kEmbedDim});
    std::fill_n(t.data<float>(), t.get_size(), fill_value);
    return ov::get_tensor_impl(t);
}

}  // namespace

TEST(ClearUnfilledAccumulatorSlotsTest, ZeroesOnlySlotsBeyondPerTokenCount) {
    // Sentinel value simulates stale data left over from a previous call, so any cell that
    // is NOT correctly zeroed by the function under test is trivially detectable.
    constexpr float kSentinel = 123.0f;
    auto acc = make_accumulator(kSentinel);

    // token 0: only 1 of 3 slots filled; token 1: 2 of 3; token 2: all 3 (no gap);
    // token 3: none filled (worst case, router filtered out every selection).
    const std::vector<size_t> token_slot_count = {1, 2, 3, 0};

    clear_unfilled_accumulator_slots(acc, token_slot_count, kNumActiveExperts, kEmbedDim, kNumTokens);

    const auto* data = acc->data<float>();
    const size_t slot_stride = kNumTokens * kEmbedDim;
    auto cell = [&](size_t slot, size_t token) {
        return data + slot * slot_stride + token * kEmbedDim;
    };

    for (size_t token_id = 0; token_id < kNumTokens; ++token_id) {
        for (size_t slot = 0; slot < kNumActiveExperts; ++slot) {
            const bool should_be_filled = slot < token_slot_count[token_id];
            const float expected = should_be_filled ? kSentinel : 0.0f;
            EXPECT_FLOAT_EQ(cell(slot, token_id)[0], expected) << "token=" << token_id << " slot=" << slot;
            EXPECT_FLOAT_EQ(cell(slot, token_id)[1], expected) << "token=" << token_id << " slot=" << slot;
        }
    }
}

TEST(ClearUnfilledAccumulatorSlotsTest, NoOpWhenEveryTokenFullyRouted) {
    constexpr float kSentinel = 7.0f;
    auto acc = make_accumulator(kSentinel);
    const std::vector<size_t> token_slot_count(kNumTokens, kNumActiveExperts);

    clear_unfilled_accumulator_slots(acc, token_slot_count, kNumActiveExperts, kEmbedDim, kNumTokens);

    const auto* data = acc->data<float>();
    for (size_t i = 0; i < acc->get_size(); ++i) {
        EXPECT_FLOAT_EQ(data[i], kSentinel);
    }
}

// End-to-end: scatter a sparse selection (one token skipped by one expert slot) into a
// buffer pre-filled with garbage, then run the targeted clear — result must match what a
// full up-front memset(0) followed by the same scatter would have produced.
TEST(ClearUnfilledAccumulatorSlotsTest, MatchesFullClearThenScatterForSparseRouting) {
    auto acc = make_accumulator(/*fill_value=*/-1.0f);

    // Only token 1 misses one expert slot (slot 2); every other (token, slot) is scattered.
    std::vector<size_t> token_slot_count(kNumTokens, 0);
    auto scatter_one = [&](size_t slot, const std::vector<size_t>& tokens, float value) {
        ov::Tensor expert_out(ov::element::f32, ov::Shape{tokens.size(), kEmbedDim});
        std::fill_n(expert_out.data<float>(), expert_out.get_size(), value);
        std::vector<size_t> slots(tokens.size(), slot);
        scatter_expert_outputs(ov::get_tensor_impl(expert_out),
                               acc,
                               tokens,
                               /*chunk_start=*/0,
                               /*chunk_size=*/tokens.size(),
                               kEmbedDim,
                               kNumTokens,
                               slots);
        for (auto token_id : tokens) {
            token_slot_count[token_id] = std::max(token_slot_count[token_id], slot + 1);
        }
    };
    scatter_one(0, {0, 1, 2, 3}, 1.0f);
    scatter_one(1, {0, 1, 2, 3}, 2.0f);
    scatter_one(2, {0, 2, 3}, 3.0f);  // token 1 skipped at slot 2

    clear_unfilled_accumulator_slots(acc, token_slot_count, kNumActiveExperts, kEmbedDim, kNumTokens);

    const auto* data = acc->data<float>();
    const size_t slot_stride = kNumTokens * kEmbedDim;
    auto value_at = [&](size_t slot, size_t token) {
        return data[slot * slot_stride + token * kEmbedDim];
    };
    EXPECT_FLOAT_EQ(value_at(0, 1), 1.0f);
    EXPECT_FLOAT_EQ(value_at(1, 1), 2.0f);
    EXPECT_FLOAT_EQ(value_at(2, 1), 0.0f);  // must be zero, not the stale -1.0f sentinel
    EXPECT_FLOAT_EQ(value_at(2, 0), 3.0f);
}
