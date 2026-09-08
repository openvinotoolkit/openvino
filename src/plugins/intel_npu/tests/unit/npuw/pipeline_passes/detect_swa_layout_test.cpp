// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "kv_cache_sliding_window_manager.hpp"
#include "npuw_transformations/detect_causal_mask.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/openvino.hpp"

using ov::npuw::util::detect_swa_layout;
using ov::npuw::util::SwaLayout;

namespace {

// Builds a minimal SDPA-per-layer model for detect_swa_layout tests.
// Entries absent from layer_mask_annotations are left unannotated.
std::shared_ptr<ov::Model> make_annotated_sdpa_model(size_t num_layers,
                                                     const std::map<size_t, int64_t>& layer_mask_annotations) {
    using namespace ov::op;
    ov::ParameterVector params;
    ov::ResultVector results;
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        auto q = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, 8});
        auto k = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, 8});
        auto v = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, 8});
        auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(q, k, v, false);
        sdpa->set_friendly_name("model.layers." + std::to_string(layer_idx) + ".self_attn/sdpa");
        const auto it = layer_mask_annotations.find(layer_idx);
        if (it != layer_mask_annotations.end()) {
            sdpa->get_rt_info()[ov::npuw::NPUW_SDPA_MASK_RT_KEY] = it->second;
        }
        params.insert(params.end(), {q, k, v});
        results.push_back(std::make_shared<v0::Result>(sdpa->output(0)));
    }
    return std::make_shared<ov::Model>(results, params);
}

}  // namespace

// No annotations -> disabled.
TEST(DetectSwaLayoutTest, NoAnnotatedLayers_IsDisabled) {
    const auto model = make_annotated_sdpa_model(2, {});
    const SwaLayout layout = detect_swa_layout(model);
    EXPECT_EQ(layout.window_size, 0u);
    EXPECT_TRUE(layout.layer_is_sliding.empty());
}

// All sliding, no full-attention -> disabled.
TEST(DetectSwaLayoutTest, AllLayersSliding_IsDisabled) {
    const auto model = make_annotated_sdpa_model(3, {{0, 128}, {1, 128}, {2, 128}});
    const SwaLayout layout = detect_swa_layout(model);
    EXPECT_EQ(layout.window_size, 0u);
    EXPECT_TRUE(layout.layer_is_sliding.empty());
}

// All causal, no sliding -> disabled.
TEST(DetectSwaLayoutTest, AllLayersCausal_IsDisabled) {
    const auto model = make_annotated_sdpa_model(3, {{0, -1}, {1, -1}, {2, -1}});
    const SwaLayout layout = detect_swa_layout(model);
    EXPECT_EQ(layout.window_size, 0u);
    EXPECT_TRUE(layout.layer_is_sliding.empty());
}

// Mixed sliding/causal -> enabled with expected pattern.
TEST(DetectSwaLayoutTest, MixedSlidingAndCausal_IsEnabledWithCorrectPattern) {
    const auto model = make_annotated_sdpa_model(4, {{0, 128}, {1, -1}, {2, 128}, {3, -1}});
    const SwaLayout layout = detect_swa_layout(model);
    ASSERT_EQ(layout.window_size, 128u);
    ASSERT_EQ(layout.layer_is_sliding.size(), 4u);
    EXPECT_TRUE(layout.layer_is_sliding[0]);
    EXPECT_FALSE(layout.layer_is_sliding[1]);
    EXPECT_TRUE(layout.layer_is_sliding[2]);
    EXPECT_FALSE(layout.layer_is_sliding[3]);
}

// Unannotated layer is treated as full-attention.
TEST(DetectSwaLayoutTest, UnannotatedLayer_TreatedAsFullAttention) {
    const auto model = make_annotated_sdpa_model(3, {{0, 64}, {2, 64}});
    const SwaLayout layout = detect_swa_layout(model);
    ASSERT_EQ(layout.window_size, 64u);
    ASSERT_EQ(layout.layer_is_sliding.size(), 3u);
    EXPECT_TRUE(layout.layer_is_sliding[0]);
    EXPECT_FALSE(layout.layer_is_sliding[1]);
    EXPECT_TRUE(layout.layer_is_sliding[2]);
}

// Sliding layers must use a single uniform window size.
TEST(DetectSwaLayoutTest, InconsistentWindowSizes_Throws) {
    const auto model = make_annotated_sdpa_model(3, {{0, 128}, {1, -1}, {2, 256}});
    EXPECT_THROW(detect_swa_layout(model), ov::Exception);
}
