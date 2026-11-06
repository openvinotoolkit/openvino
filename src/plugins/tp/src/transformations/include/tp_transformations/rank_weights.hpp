// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <vector>

#include "openvino/core/model.hpp"

namespace ov {
namespace tp {

/// The weights of the rank models, read off the analyzed model before any of them is built.
///
/// A lower bound of what a device holds: a constant cut at compile time or behind a runtime Slice
/// counts its rank's share, and an f32 / f64 constant counts two bytes per element, since a device
/// may keep it in half precision.
struct RankWeights {
    std::vector<size_t> per_rank;  ///< bytes of rank r's constants
    size_t replicated = 0;         ///< bytes every rank keeps whole
    size_t split = 0;              ///< bytes of the constants the ranks share out
};

RankWeights rank_weights(const ov::Model& analyzed, size_t world_size);

/// Fewest ranks whose share of the weights fits in `free_bytes`, splitting `split` evenly;
/// 0 when the replicated weights alone do not fit.
size_t min_world_size(const RankWeights& weights, size_t free_bytes);

}  // namespace tp
}  // namespace ov
