// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "pipelines/kokoro/kokoro_utils.hpp"

using ov::npuw::kokoro::fill_text_mask_from_lengths;
using ov::npuw::kokoro::find_real_sequence_length;
using ov::npuw::kokoro::zero_padding_durations;

// ============================================================================
// fill_text_mask end-to-end: input_ids → real_len → mask
// This mirrors the logic in KokoroInferRequest::fill_text_mask().
// ============================================================================

// Typical Kokoro input: [BOS=0, phoneme1..N, EOS=0, PAD...]
TEST(KokoroPadding, FillTextMask_TypicalPaddedInput) {
    std::vector<int64_t> ids{0, 10, 20, 30, 0, 0, 0, 0};  // 3 phonemes + padding
    const std::size_t seq_len = ids.size();

    const auto real_len = find_real_sequence_length(ids.data(), seq_len);
    EXPECT_EQ(real_len, 5u);  // BOS + 3 phonemes + EOS

    std::vector<bool> mask(seq_len);
    fill_text_mask_from_lengths(mask.data(), seq_len, real_len);

    // Valid positions (BOS, tok, tok, tok, EOS) → false
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_FALSE(mask[i]) << "Position " << i << " is valid, should be false";
    }
    // Padding positions → true
    for (std::size_t i = 5; i < seq_len; ++i) {
        EXPECT_TRUE(mask[i]) << "Position " << i << " is padding, should be true";
    }
}

// No padding — input fills the entire static buffer
TEST(KokoroPadding, FillTextMask_NoPadding) {
    std::vector<int64_t> ids{0, 1, 2, 3, 4, 5};  // all non-zero after BOS
    const std::size_t seq_len = ids.size();

    const auto real_len = find_real_sequence_length(ids.data(), seq_len);
    EXPECT_EQ(real_len, seq_len);  // no EOS found → entire buffer is valid

    std::vector<bool> mask(seq_len);
    fill_text_mask_from_lengths(mask.data(), seq_len, real_len);

    for (std::size_t i = 0; i < seq_len; ++i) {
        EXPECT_FALSE(mask[i]);
    }
}

// Empty phoneme sequence: [BOS=0, EOS=0, PAD, PAD, ...]
TEST(KokoroPadding, FillTextMask_EmptyPhonemes) {
    std::vector<int64_t> ids{0, 0, 0, 0};
    const auto real_len = find_real_sequence_length(ids.data(), ids.size());
    EXPECT_EQ(real_len, 2u);  // BOS + EOS only

    std::vector<bool> mask(ids.size());
    fill_text_mask_from_lengths(mask.data(), ids.size(), real_len);

    EXPECT_FALSE(mask[0]);  // BOS
    EXPECT_FALSE(mask[1]);  // EOS
    EXPECT_TRUE(mask[2]);   // PAD
    EXPECT_TRUE(mask[3]);   // PAD
}

// Non-zero padding tokens after EOS: [BOS=0, tok, EOS=0, 20, 20, 20]
// User might accidentally fill padding with a real token id (as it was before with " ")
TEST(KokoroPadding, FillTextMask_NonZeroPaddingAfterEos) {
    std::vector<int64_t> ids{0, 42, 0, 20, 20, 20};
    const std::size_t seq_len = ids.size();

    const auto real_len = find_real_sequence_length(ids.data(), seq_len);
    EXPECT_EQ(real_len, 3u);  // BOS + 1 phoneme + EOS

    std::vector<bool> mask(seq_len);
    fill_text_mask_from_lengths(mask.data(), seq_len, real_len);

    EXPECT_FALSE(mask[0]);  // BOS
    EXPECT_FALSE(mask[1]);  // tok
    EXPECT_FALSE(mask[2]);  // EOS
    EXPECT_TRUE(mask[3]);   // non-zero padding — still masked
    EXPECT_TRUE(mask[4]);
    EXPECT_TRUE(mask[5]);
}

// ============================================================================
// zero_padding_durations: pred_dur truncation
// ============================================================================

TEST(KokoroPadding, ZeroPadDurations_PaddingZeroed) {
    std::vector<int64_t> ids{0, 10, 20, 0, 0, 0};
    const auto real_len = find_real_sequence_length(ids.data(), ids.size());

    // Simulate pred_dur output — every position got a non-zero duration 
    std::vector<int64_t> dur{3, 5, 2, 4, 8, 1};
    zero_padding_durations(dur.data(), dur.size(), real_len);

    // Valid positions preserved
    EXPECT_EQ(dur[0], 3);
    EXPECT_EQ(dur[1], 5);
    EXPECT_EQ(dur[2], 2);
    EXPECT_EQ(dur[3], 4);
    // Padding zeroed
    EXPECT_EQ(dur[4], 0);
    EXPECT_EQ(dur[5], 0);
}

