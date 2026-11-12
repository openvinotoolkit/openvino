// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tp_transformations/rank_weights.hpp"

#include <numeric>
#include <unordered_map>

#include "openvino/op/constant.hpp"
#include "tp_transformations/annotations.hpp"

namespace ov {
namespace tp {
namespace {

size_t device_bytes(const ov::op::v0::Constant& constant) {
    const auto type = constant.get_element_type();
    if (type == ov::element::f32 || type == ov::element::f64)
        return ov::shape_size(constant.get_shape()) * 2;
    return constant.get_byte_size();
}

}  // namespace

RankWeights rank_weights(const ov::Model& analyzed, size_t world_size) {
    // The constants each rank holds a share of, with the annotation that says which.
    std::unordered_map<const ov::Node*, const ShardAnnotation*> split;
    for (const auto& node : analyzed.get_ops()) {
        const auto* annotation = get_shard(*node);
        if (annotation && annotation->elements.empty() && ov::is_type<ov::op::v0::Constant>(node))
            split.emplace(node.get(), annotation);
    }

    RankWeights weights;
    weights.per_rank.assign(world_size, 0);
    for (const auto& node : analyzed.get_ops()) {
        const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
        if (!constant)
            continue;
        const size_t bytes = device_bytes(*constant);
        const auto it = split.find(node.get());
        if (it == split.end()) {
            weights.replicated += bytes;
            continue;
        }
        weights.split += bytes;
        const auto& segments = it->second->segments;
        const size_t total = std::accumulate(segments.begin(), segments.end(), size_t{0}, [](size_t sum, const Segment& s) {
            return sum + s.length;
        });
        for (size_t rank = 0; rank < world_size; ++rank)
            weights.per_rank[rank] += total == 0 ? bytes : bytes / total * rank_length(segments, rank, world_size) +
                                                               bytes % total * rank_length(segments, rank, world_size) / total;
    }
    for (auto& bytes : weights.per_rank)
        bytes += weights.replicated;
    return weights;
}

size_t min_world_size(const RankWeights& weights, size_t free_bytes) {
    if (weights.replicated >= free_bytes)
        return 0;
    const size_t room = free_bytes - weights.replicated;
    return std::max<size_t>(1, (weights.split + room - 1) / room);
}

}  // namespace tp
}  // namespace ov
