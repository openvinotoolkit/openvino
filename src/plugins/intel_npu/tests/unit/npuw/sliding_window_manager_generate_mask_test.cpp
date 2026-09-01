// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "kv_cache_sliding_window_manager.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace ov::test::npuw {

namespace {

namespace uu = ov::npuw::util;

ov::SoPtr<ov::ITensor> make_mask_tensor(uint32_t rows, uint32_t cols, float init_value) {
    auto mask = ov::get_tensor_impl(ov::Tensor(ov::element::f32, ov::Shape{1u, rows, cols}));
    std::fill(mask->data<float>(), mask->data<float>() + mask->get_size(), init_value);
    return mask;
}

float mask_at(const ov::SoPtr<ov::ITensor>& mask, uint32_t row, uint32_t col) {
    const auto& shape = mask->get_shape();
    const auto rows = static_cast<uint32_t>(shape[shape.size() - 2]);
    const auto cols = static_cast<uint32_t>(shape[shape.size() - 1]);
    OPENVINO_ASSERT(row < rows && col < cols, "mask_at index out of range");
    return mask->data<float>()[static_cast<size_t>(row) * cols + col];
}

}  // namespace

class FillCausalSlidingMaskTest : public ::testing::Test {};

TEST_F(FillCausalSlidingMaskTest, UnsaturatedPastBuildsExpectedMask) {
    auto mask = make_mask_tensor(/*rows=*/4u, /*cols=*/8u, /*init_value=*/777.f);
    const float kMasked = static_cast<float>(std::numeric_limits<ov::float16>::lowest());

    uu::fill_causal_sliding_mask(mask,
                                 /*num_stored_tokens_before=*/2u,
                                 /*num_real_new_tokens=*/2u,
                                 /*window_size=*/3u);

    // row=2: q=2, past abs=[0,1] visible; only diagonal local key is visible.
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 0u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 1u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 2u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 3u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 6u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 7u), kMasked);

    // row=3: abs=0 is out-of-window, abs=1 is still visible.
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 0u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 1u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 6u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 7u), 0.f);
}

TEST_F(FillCausalSlidingMaskTest, SaturatedPastUsesCircularSlotMapping) {
    auto mask = make_mask_tensor(/*rows=*/2u, /*cols=*/6u, /*init_value=*/777.f);
    const float kMasked = static_cast<float>(std::numeric_limits<ov::float16>::lowest());

    uu::fill_causal_sliding_mask(mask,
                                 /*num_stored_tokens_before=*/6u,
                                 /*num_real_new_tokens=*/2u,
                                 /*window_size=*/3u);

    // past_width=4, r=2 => slot->abs=[4,5,2,3]; row=0(q=6): 4/5 visible, 2/3 masked.
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 0u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 1u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 2u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 3u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 4u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 5u), kMasked);

    // row=1(q=7): abs 5 visible, abs 4 masked.
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 0u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 1u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 4u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 5u), 0.f);
}

TEST_F(FillCausalSlidingMaskTest, ZeroPastWidthFallsBackToCurrentChunkSlidingCausalMask) {
    // rows == cols => past_width == 0, so mask must be built from current chunk only.
    auto mask = make_mask_tensor(/*rows=*/4u, /*cols=*/4u, /*init_value=*/777.f);
    const float kMasked = static_cast<float>(std::numeric_limits<ov::float16>::lowest());

    uu::fill_causal_sliding_mask(mask,
                                 /*num_stored_tokens_before=*/0u,
                                 /*num_real_new_tokens=*/4u,
                                 /*window_size=*/2u);

    // Expected visible local columns per row for window=2:
    // row0 -> [0]
    // row1 -> [0,1]
    // row2 -> [1,2]
    // row3 -> [2,3]
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 0u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 1u), kMasked);

    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 0u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 1u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 2u), kMasked);

    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 0u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 1u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 2u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 3u), kMasked);

    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 1u), kMasked);
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 2u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 3u), 0.f);
}

class OverlayVisionBidirectionalMaskTest : public ::testing::Test {};

TEST_F(OverlayVisionBidirectionalMaskTest, SingleVisionRunUnmasksRunBlock) {
    auto mask = make_mask_tensor(/*rows=*/4u, /*cols=*/8u, /*init_value=*/-7.f);
    const std::vector<int64_t> token_types = {0, 1, 1, 0};

    uu::overlay_vision_bidirectional_mask(mask, token_types.data(), static_cast<uint32_t>(token_types.size()));

    // run [1,3): rows 1..2 and cols 5..6 are unmasked.
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 5u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 6u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 5u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 6u), 0.f);

    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 4u), -7.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 5u), -7.f);
}

TEST_F(OverlayVisionBidirectionalMaskTest, DisjointVisionRunsAreHandledSeparately) {
    auto mask = make_mask_tensor(/*rows=*/4u, /*cols=*/8u, /*init_value=*/-9.f);
    const std::vector<int64_t> token_types = {1, 1, 0, 1};

    uu::overlay_vision_bidirectional_mask(mask, token_types.data(), static_cast<uint32_t>(token_types.size()));

    // run [0,2): rows 0..1 and cols 4..5.
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 4u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 0u, 5u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 4u), 0.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 1u, 5u), 0.f);

    // run [3,4): row 3 and col 7.
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 7u), 0.f);

    EXPECT_FLOAT_EQ(mask_at(mask, 2u, 7u), -9.f);
    EXPECT_FLOAT_EQ(mask_at(mask, 3u, 6u), -9.f);
}

}  // namespace ov::test::npuw
