// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "turboq_rotation.hpp"

#include <cstdint>
#include <map>
#include <random>
#include <vector>

namespace ov::Extensions::Cpu {

static constexpr uint64_t TURBOQ_SEED = 0x517cc1b727220a95ULL;
static constexpr uint64_t TURBOQ_WHT_SEED = TURBOQ_SEED ^ 0xfedcba9876543210ULL;

// Per-thread, per-dimension sign cache.
const float* turboq_get_wht_signs(int dim) {
    thread_local std::map<int, std::vector<float>> cache;
    auto [it, inserted] = cache.try_emplace(dim);
    if (!inserted) {
        return it->second.data();
    }
    auto& signs = it->second;
    signs.resize(dim);
    std::mt19937_64 rng(TURBOQ_WHT_SEED);
    std::uniform_int_distribution<int> dist(0, 1);
    for (int i = 0; i < dim; i++) {
        signs[i] = dist(rng) ? 1.0F : -1.0F;
    }
    return signs.data();
}

}  // namespace ov::Extensions::Cpu
