// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>

#include "openvino/core/except.hpp"
#include "openvino/core/partial_shape.hpp"

namespace ov::intel_gpu::op {

// Physical layout of FullyConnected weights.
enum class FcWeightsLayout : uint8_t {
    NK,   // [..., N, K]
    KN,   // [..., K, N]
    GNK,  // [K / gs, N, gs]
};

// Logical weight dims: out = A * W^T with W [N, K].
struct FcWeightDims {
    ov::Dimension N;
    ov::Dimension K;
    int64_t G = 1;   // number of K groups (GNK only)
    int64_t gs = 0;  // group size (GNK only)
};

inline FcWeightsLayout fc_default_weights_layout(bool transpose_b) {
    return transpose_b ? FcWeightsLayout::NK : FcWeightsLayout::KN;
}

// GNK implies transposed weights; NK / KN must match transpose_b.
inline bool fc_weights_layout_is_consistent(FcWeightsLayout layout, bool transpose_b) {
    return layout == FcWeightsLayout::GNK ? transpose_b : layout == fc_default_weights_layout(transpose_b);
}

// Single source of truth for logical N / K given the physical weight shape and layout.
inline FcWeightDims fc_weight_dims(const ov::PartialShape& w, FcWeightsLayout layout) {
    OPENVINO_ASSERT(w.rank().is_static() && w.size() >= 2, "[GPU] FullyConnected weights need a static rank >= 2, got ", w);
    const auto r = w.size();
    switch (layout) {
    case FcWeightsLayout::NK:
        return {w[r - 2], w[r - 1]};
    case FcWeightsLayout::KN:
        return {w[r - 1], w[r - 2]};
    case FcWeightsLayout::GNK: {
        OPENVINO_ASSERT(r == 3 && w[0].is_static() && w[2].is_static(), "[GPU] GNK FullyConnected weights must be [G, N, gs] with static G and gs, got ", w);
        const auto G = w[0].get_length();
        const auto gs = w[2].get_length();
        return {w[1], w[0] * w[2], G, gs};
    }
    }
    OPENVINO_THROW("[GPU] Unknown FullyConnected weights layout");
}

inline const char* to_string(FcWeightsLayout layout) {
    switch (layout) {
    case FcWeightsLayout::NK:
        return "NK";
    case FcWeightsLayout::KN:
        return "KN";
    case FcWeightsLayout::GNK:
        return "GNK";
    }
    OPENVINO_THROW("[GPU] Unknown FullyConnected weights layout");
}

inline FcWeightsLayout fc_weights_layout_from_string(const std::string& s) {
    if (s == "NK")
        return FcWeightsLayout::NK;
    if (s == "KN")
        return FcWeightsLayout::KN;
    if (s == "GNK")
        return FcWeightsLayout::GNK;
    OPENVINO_THROW("[GPU] Unknown FullyConnected weights layout: ", s);
}

}  // namespace ov::intel_gpu::op
