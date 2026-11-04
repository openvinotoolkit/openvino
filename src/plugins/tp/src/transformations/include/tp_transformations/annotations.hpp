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
///
/// Rank r owns share (r / step) % count of the segment's units, count 0 meaning world_size: step R
/// gives R consecutive ranks the same share (KV heads shared by fewer heads than ranks), count R
/// splits the segment in R shares repeated over the ranks (a KV head's query group).
struct Segment {
    size_t length = 0;  ///< elements along the split dimension
    size_t unit = 1;    ///< split granularity: a head (group), a quantization group, or 1
    size_t step = 1;
    size_t count = 0;

    bool operator==(const Segment& other) const {
        return length == other.length && unit == other.unit && step == other.step && count == other.count;
    }
};

enum class ShardStrategy { COLUMN_PARALLEL, ROW_PARALLEL, REPLICATED };

/// The collective a projection's output goes through.
enum class Collective { NONE, ALL_REDUCE, GATHER };

/// rt_info["tp_shard"]: how a rank-dependent node is cut per rank.
///
/// On a Constant without `elements`: the tensor is sliced along `axis`.
/// On a Constant with `elements`: a shape constant whose listed elements get rank values, as
/// `element_mode` says.
/// On a ReadValue: its variable's shape is cut along `axis`.
/// On any node with `input` >= 0: that input is sliced along `axis` at run time
/// (a weight chain that cannot be cut at compile time).
/// On any node with `rt_keys`: those rt_info entries hold the global size and
/// get the rank's size.
class ShardAnnotation : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("tp_shard", "0", ov::RuntimeAttribute);

    /// What the listed elements of a shape constant hold.
    enum class Elements {
        SIZE,           ///< the split dimension's size: the rank's size of all segments
        SEGMENT_SIZES,  ///< listed element i: the size of segment i (split lengths of a fused projection)
        BOUNDS,         ///< a boundary between segments: the rank's boundary (slice bounds of a fused projection)
    };

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
    Elements element_mode = Elements::SIZE;
    std::vector<std::string> rt_keys;
    int64_t input = -1;

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
    ProjectionRole(ShardStrategy strategy, Collective collective, int group_id, std::string block)
        : strategy(strategy),
          collective(collective),
          group_id(group_id),
          block(std::move(block)) {}

    ShardStrategy strategy = ShardStrategy::REPLICATED;
    Collective collective = Collective::NONE;  ///< inserted on the MatMul's output
    int group_id = -1;                         ///< collective group; -1 with no collective
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

/// The rank's value of the `listed`-th element of an annotated shape constant whose global value is `global`.
int64_t localize(const ShardAnnotation& annotation, size_t listed, int64_t global, size_t rank, size_t world_size);

}  // namespace tp
}  // namespace ov
