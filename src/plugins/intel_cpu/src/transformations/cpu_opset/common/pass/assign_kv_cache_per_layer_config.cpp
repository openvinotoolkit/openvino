// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "assign_kv_cache_per_layer_config.hpp"

#include <cstddef>
#include <memory>

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "transformations/cpu_opset/common/op/sdpa.hpp"

namespace ov::intel_cpu {

namespace {
bool is_sdpa_like(const std::shared_ptr<ov::Node>& op) {
    return ov::is_type<ScaledDotProductAttentionWithKVCache>(op) ||
           ov::is_type<SDPAWithTransposeReshape>(op) ||
           ov::is_type<ov::op::v13::ScaledDotProductAttention>(op);
}
}  // namespace

bool AssignKVCachePerLayerConfig::run_on_model(const std::shared_ptr<ov::Model>& model) {
    if (m_per_layer_config.empty()) {
        return false;
    }

    size_t sdpa_count = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (is_sdpa_like(op)) {
            ++sdpa_count;
        }
    }

    OPENVINO_ASSERT(sdpa_count == m_per_layer_config.size(),
                    "KV_CACHE_PER_LAYER_CONFIG length ",
                    m_per_layer_config.size(),
                    " does not match SDPA layer count ",
                    sdpa_count,
                    " in the model.");

    size_t idx = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (!is_sdpa_like(op)) {
            continue;
        }
        auto& rt = op->get_rt_info();
        rt["kv_cache_layer_idx"] = idx;
        rt["kv_cache_layer_config"] = m_per_layer_config[idx];
        ++idx;
    }
    return true;
}

}  // namespace ov::intel_cpu
