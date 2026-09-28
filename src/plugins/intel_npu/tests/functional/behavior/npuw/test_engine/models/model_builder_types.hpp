// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Common typedefs and small structs shared across all model_builder_* headers.
//

#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/op/util/variable.hpp"

namespace ov {
namespace test {
namespace npuw {

struct KVCacheResult {
    ov::Output<ov::Node> concatenated;
    ov::Output<ov::Node> beam_gather;
    std::shared_ptr<ov::Node> assign;
};

struct KVCacheReadState {
    std::shared_ptr<ov::op::util::Variable> variable;
    ov::Output<ov::Node> beam_gather;
};

using WeightFn = std::function<ov::Output<ov::Node>(const std::string&, const ov::Shape&, ov::element::Type)>;
using NormFn = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, const std::string&)>;
using FFNFn = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, const std::string&)>;
using RoPEFn = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, const std::string&)>;
using LayerFn = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, const std::string&, size_t)>;

/// (projected_k, projected_v, layer_idx) -> (cached_k, cached_v). Empty = no cache.
using KVCacheFn =
    std::function<std::pair<ov::Output<ov::Node>,
                            ov::Output<ov::Node>>(const ov::Output<ov::Node>&, const ov::Output<ov::Node>&, size_t)>;

/// LoRA injection state: when passed (non-null) to make_linear, the projection
/// gets three extra A/B/alpha tensors wired as
///   output = base_linear_out + (input @ A^T * alpha) @ B^T
/// Names follow the NPUW lora_state_* convention.
struct LoRAInjector {
    size_t max_rank = 0;
    std::vector<std::string> targets;
    ov::element::Type precision = ov::element::f32;
    bool stateful = false;
    ov::SinkVector* sinks = nullptr;

    static const std::vector<std::string>& default_targets() {
        static const std::vector<std::string> defaults =
            {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"};
        return defaults;
    }

    bool should_adapt(const std::string& name) const {
        if (max_rank == 0) {
            return false;
        }
        const auto& list = targets.empty() ? default_targets() : targets;
        return std::any_of(list.begin(), list.end(), [&name](const std::string& t) {
            return name.find(t) != std::string::npos;
        });
    }
};

}  // namespace npuw
}  // namespace test
}  // namespace ov
