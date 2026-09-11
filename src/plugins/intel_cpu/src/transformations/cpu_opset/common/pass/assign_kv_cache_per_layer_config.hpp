// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_cpu {

// Walks the model in topological order, finds SDPA-like ops
// (ScaledDotProductAttentionWithKVCache, ScaledDotProductAttention v13,
// PagedAttentionExtension), and attaches the matching positional entry from
// `per_layer_config` as rt_info["kv_cache_layer_config"]. Throws if vector
// length does not match the SDPA layer count.
class AssignKVCachePerLayerConfig : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("AssignKVCachePerLayerConfig");
    explicit AssignKVCachePerLayerConfig(std::vector<ov::AnyMap> per_layer_config)
        : m_per_layer_config(std::move(per_layer_config)) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::vector<ov::AnyMap> m_per_layer_config;
};

}  // namespace ov::intel_cpu
