// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Common typedefs and small structs shared across all model_builder_* headers.
//

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
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

}  // namespace npuw
}  // namespace test
}  // namespace ov
