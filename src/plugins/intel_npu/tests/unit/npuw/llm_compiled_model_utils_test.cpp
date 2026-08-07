// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <map>

#include "llm_compiled_model_utils.hpp"

using ov::npuw::util::derive_swa_layout;
using ov::npuw::util::SwaLayout;

// Empty annotations (no SDPA nodes recognized at all) -> disabled.
TEST(DeriveSwaLayoutTest, EmptyAnnotations_IsDisabled) {
    const SwaLayout layout = derive_swa_layout({});
    EXPECT_EQ(layout.window_size, 0u);
    EXPECT_TRUE(layout.layer_is_sliding.empty());
}

// Every layer sliding, none full-attention -> not a "genuine hybrid" model, stays disabled.
TEST(DeriveSwaLayoutTest, AllLayersSliding_IsDisabled) {
    const std::map<size_t, int64_t> annotations = {{0, 128}, {1, 128}, {2, 128}};
    const SwaLayout layout = derive_swa_layout(annotations);
    EXPECT_EQ(layout.window_size, 0u);
    EXPECT_TRUE(layout.layer_is_sliding.empty());
}

// Every layer causal (negative encoding), none sliding -> stays disabled.
TEST(DeriveSwaLayoutTest, AllLayersCausal_IsDisabled) {
    const std::map<size_t, int64_t> annotations = {{0, -1}, {1, -1}, {2, -1}};
    const SwaLayout layout = derive_swa_layout(annotations);
    EXPECT_EQ(layout.window_size, 0u);
    EXPECT_TRUE(layout.layer_is_sliding.empty());
}

// Genuine hybrid: some sliding, some causal -> enabled with the right per-layer pattern.
TEST(DeriveSwaLayoutTest, MixedSlidingAndCausal_IsEnabledWithCorrectPattern) {
    const std::map<size_t, int64_t> annotations = {{0, 128}, {1, -1}, {2, 128}, {3, -1}};
    const SwaLayout layout = derive_swa_layout(annotations);
    ASSERT_EQ(layout.window_size, 128u);
    ASSERT_EQ(layout.layer_is_sliding.size(), 4u);
    EXPECT_TRUE(layout.layer_is_sliding[0]);
    EXPECT_FALSE(layout.layer_is_sliding[1]);
    EXPECT_TRUE(layout.layer_is_sliding[2]);
    EXPECT_FALSE(layout.layer_is_sliding[3]);
}

// A layer index missing from the map (mask pattern not recognized, i.e. Unknown) is treated as
// full-attention, same as an explicit negative/causal entry.
TEST(DeriveSwaLayoutTest, MissingLayerIndex_TreatedAsFullAttention) {
    // Layer 1 is absent entirely (Unknown); layers 0 and 2 are sliding.
    const std::map<size_t, int64_t> annotations = {{0, 64}, {2, 64}};
    const SwaLayout layout = derive_swa_layout(annotations);
    ASSERT_EQ(layout.window_size, 64u);
    ASSERT_EQ(layout.layer_is_sliding.size(), 3u);
    EXPECT_TRUE(layout.layer_is_sliding[0]);
    EXPECT_FALSE(layout.layer_is_sliding[1]);
    EXPECT_TRUE(layout.layer_is_sliding[2]);
}

// Sliding layers must agree on a single, uniform window size.
TEST(DeriveSwaLayoutTest, InconsistentWindowSizes_Throws) {
    const std::map<size_t, int64_t> annotations = {{0, 128}, {1, -1}, {2, 256}};
    EXPECT_THROW(derive_swa_layout(annotations), ov::Exception);
}
