// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"

namespace ov {
class Model;

namespace tp {

/// A part of a split dimension that is split on its own (one per fused segment).
struct Segment {
    size_t length = 0;  ///< elements along the split dimension
    size_t unit = 1;    ///< split granularity: a head (group), a quantization group, or 1

    bool operator==(const Segment& other) const {
        return length == other.length && unit == other.unit;
    }
};

enum class ShardStrategy { COLUMN_PARALLEL, ROW_PARALLEL, REPLICATED };

/// rt_info["tp_shard"]: how a rank-dependent node is cut per rank.
///
/// On a Constant without `elements`: the tensor is sliced along `axis`.
/// On a Constant with `elements`: a shape constant; those elements hold the
/// split dimension's global size and get the rank's size.
/// On any node with `rt_keys`: those rt_info entries hold the global size and
/// get the rank's size.
class ShardAnnotation : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("tp_shard", "0", ov::RuntimeAttribute);

    ShardAnnotation() = default;
    ShardAnnotation(ShardStrategy strategy, int64_t axis, std::vector<Segment> segments, std::string block)
        : strategy(strategy),
          axis(axis),
          segments(std::move(segments)),
          block(std::move(block)) {}

    ShardStrategy strategy = ShardStrategy::REPLICATED;
    int64_t axis = -1;
    std::vector<Segment> segments;
    std::string block;  ///< block pass that set it
    std::vector<size_t> elements;
    std::vector<std::string> rt_keys;

    bool is_copyable() const override {
        return true;
    }
    bool is_deterministic() const override {
        return false;
    }
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    std::string to_string() const override;
    bool operator==(const ShardAnnotation& other) const;
};

/// rt_info["tp_role"]: the role of a projection MatMul.
class ProjectionRole : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("tp_role", "0", ov::RuntimeAttribute);

    ProjectionRole() = default;
    ProjectionRole(ShardStrategy strategy, int allreduce_group_id, std::string block)
        : strategy(strategy),
          allreduce_group_id(allreduce_group_id),
          block(std::move(block)) {}

    ShardStrategy strategy = ShardStrategy::REPLICATED;
    int allreduce_group_id = -1;  ///< group of the AllReduce on its output; -1: none
    std::string block;

    bool is_copyable() const override {
        return true;
    }
    bool is_deterministic() const override {
        return false;
    }
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    std::string to_string() const override;
    bool operator==(const ProjectionRole& other) const;
};

/// Sets the annotation; throws naming both blocks when the node already has a different one.
void set_shard(const std::shared_ptr<ov::Node>& node, const ShardAnnotation& annotation);
void set_role(const std::shared_ptr<ov::Node>& node, const ProjectionRole& role);

const ShardAnnotation* get_shard(const ov::Node& node);
const ProjectionRole* get_role(const ov::Node& node);

/// Removes both annotations from every node of the model.
void strip_annotations(const ov::Model& model);

/// A range of the split dimension owned by one rank.
struct Range {
    size_t offset = 0;
    size_t length = 0;

    bool operator==(const Range& other) const {
        return offset == other.offset && length == other.length;
    }
};

/// The ranges of `rank`, one per segment, offsets relative to the whole dimension.
///
/// Each segment is split in whole units; a remainder goes to the first ranks,
/// one extra unit each.
std::vector<Range> rank_ranges(const std::vector<Segment>& segments, size_t rank, size_t world_size);

/// Total length of `rank`'s ranges.
size_t rank_length(const std::vector<Segment>& segments, size_t rank, size_t world_size);

}  // namespace tp
}  // namespace ov
