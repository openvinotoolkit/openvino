// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ov {
class Model;
}  // namespace ov

namespace ov {
namespace npuw {
namespace util {

// Hybrid SWA layout descriptor.
// window_size == 0 means SWA is disabled.
struct SwaLayout {
    uint32_t window_size = 0;            // 0 == Sliding Window Attention disabled
    std::vector<bool> layer_is_sliding;  // per-layer flag, indexed by decoder layer id

    bool enabled() const {
        return window_size > 0;
    }

    // True if SWA is enabled and layer_idx is configured as a sliding-window layer.
    bool is_sliding(size_t layer_idx) const {
        return enabled() && layer_idx < layer_is_sliding.size() && layer_is_sliding[layer_idx];
    }
};

// rt_info key stamped on SWA-managed past_key_values Parameters whose seq_len
// axis was shrunk to the SWA window size.
static constexpr const char* NPUW_KV_CACHE_SLIDING_RT_KEY = "npuw_kv_cache_sliding";

// Derives hybrid SWA layout from per-layer SDPA mask annotations.
// Enables SWA only for genuine hybrid models: at least one sliding layer and one
// full/causal layer. Throws if sliding layers use different window sizes.
SwaLayout detect_swa_layout(const std::shared_ptr<ov::Model>& model);

}  // namespace util
}  // namespace npuw
}  // namespace ov
