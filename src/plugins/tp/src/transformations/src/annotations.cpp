// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tp_transformations/annotations.hpp"

#include <algorithm>
#include <sstream>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace tp {

namespace {

const char* to_cstr(ShardStrategy strategy) {
    switch (strategy) {
    case ShardStrategy::COLUMN_PARALLEL:
        return "COLUMN_PARALLEL";
    case ShardStrategy::ROW_PARALLEL:
        return "ROW_PARALLEL";
    case ShardStrategy::REPLICATED:
        return "REPLICATED";
    }
    return "?";
}

ShardStrategy from_string(const std::string& name) {
    if (name == "COLUMN_PARALLEL")
        return ShardStrategy::COLUMN_PARALLEL;
    if (name == "ROW_PARALLEL")
        return ShardStrategy::ROW_PARALLEL;
    if (name == "REPLICATED")
        return ShardStrategy::REPLICATED;
    OPENVINO_THROW("[TP] unknown shard strategy ", name);
}

const char* to_cstr(Collective collective) {
    switch (collective) {
    case Collective::NONE:
        return "NONE";
    case Collective::ALL_REDUCE:
        return "ALL_REDUCE";
    case Collective::GATHER:
        return "GATHER";
    }
    return "?";
}

Collective collective_from_string(const std::string& name) {
    if (name == "NONE")
        return Collective::NONE;
    if (name == "ALL_REDUCE")
        return Collective::ALL_REDUCE;
    if (name == "GATHER")
        return Collective::GATHER;
    OPENVINO_THROW("[TP] unknown collective ", name);
}

template <class T>
void set_annotation(const std::shared_ptr<ov::Node>& node, const T& value) {
    auto& rt_info = node->get_rt_info();
    const auto it = rt_info.find(T::get_type_info_static());
    if (it != rt_info.end()) {
        const auto& existing = it->second.template as<T>();
        OPENVINO_ASSERT(existing == value,
                        "[TP] '",
                        node->get_friendly_name(),
                        "' (",
                        node->get_type_name(),
                        ") is claimed twice: ",
                        existing.to_string(),
                        " and ",
                        value.to_string());
        return;
    }
    rt_info[T::get_type_info_static()] = value;
}

template <class T>
const T* get_annotation(const ov::Node& node) {
    const auto& rt_info = node.get_rt_info();
    const auto it = rt_info.find(T::get_type_info_static());
    return it == rt_info.end() ? nullptr : &it->second.template as<T>();
}

}  // namespace

bool ShardAnnotation::visit_attributes(ov::AttributeVisitor& visitor) {
    auto strategy_name = std::string(to_cstr(strategy));
    std::vector<int64_t> lengths, units, steps, counts, element_list;
    for (const auto& segment : segments) {
        lengths.push_back(static_cast<int64_t>(segment.length));
        units.push_back(static_cast<int64_t>(segment.unit));
        steps.push_back(static_cast<int64_t>(segment.step));
        counts.push_back(static_cast<int64_t>(segment.count));
    }
    for (const auto element : elements) {
        element_list.push_back(static_cast<int64_t>(element));
    }
    auto mode = static_cast<int64_t>(element_mode);
    visitor.on_attribute("strategy", strategy_name);
    visitor.on_attribute("axis", axis);
    visitor.on_attribute("lengths", lengths);
    visitor.on_attribute("units", units);
    visitor.on_attribute("steps", steps);
    visitor.on_attribute("counts", counts);
    visitor.on_attribute("elements", element_list);
    visitor.on_attribute("element_mode", mode);
    visitor.on_attribute("rt_keys", rt_keys);
    visitor.on_attribute("input", input);
    visitor.on_attribute("block", block);

    strategy = from_string(strategy_name);
    OPENVINO_ASSERT(lengths.size() == units.size() && lengths.size() == steps.size() && lengths.size() == counts.size(),
                    "[TP] tp_shard: ", lengths.size(), " lengths, ", units.size(), " units, ", steps.size(), " steps, ",
                    counts.size(), " counts");
    segments.clear();
    for (size_t i = 0; i < lengths.size(); ++i) {
        segments.push_back({static_cast<size_t>(lengths[i]),
                            static_cast<size_t>(units[i]),
                            static_cast<size_t>(steps[i]),
                            static_cast<size_t>(counts[i])});
    }
    elements.assign(element_list.begin(), element_list.end());
    OPENVINO_ASSERT(mode >= 0 && mode <= static_cast<int64_t>(Elements::BOUNDS), "[TP] tp_shard: element mode ", mode);
    element_mode = static_cast<Elements>(mode);
    return true;
}

std::string ShardAnnotation::to_string() const {
    std::ostringstream out;
    out << to_cstr(strategy) << " axis " << axis << " segments [";
    for (size_t i = 0; i < segments.size(); ++i) {
        out << (i ? ", " : "") << segments[i].length << "/" << segments[i].unit;
        if (segments[i].step != 1)
            out << " step " << segments[i].step;
        if (segments[i].count != 0)
            out << " count " << segments[i].count;
    }
    out << "]";
    if (!elements.empty()) {
        out << " elements [";
        for (size_t i = 0; i < elements.size(); ++i) {
            out << (i ? ", " : "") << elements[i];
        }
        out << "]";
        if (element_mode == Elements::SEGMENT_SIZES)
            out << " (segment sizes)";
        else if (element_mode == Elements::BOUNDS)
            out << " (segment bounds)";
    }
    for (const auto& key : rt_keys) {
        out << " rt_info '" << key << "'";
    }
    if (input >= 0) {
        out << " runtime slice of input " << input;
    }
    out << " by " << block;
    return out.str();
}

bool ShardAnnotation::operator==(const ShardAnnotation& other) const {
    return strategy == other.strategy && axis == other.axis && segments == other.segments &&
           elements == other.elements && element_mode == other.element_mode && rt_keys == other.rt_keys &&
           input == other.input;
}

bool ProjectionRole::visit_attributes(ov::AttributeVisitor& visitor) {
    auto strategy_name = std::string(to_cstr(strategy));
    auto collective_name = std::string(to_cstr(collective));
    visitor.on_attribute("strategy", strategy_name);
    visitor.on_attribute("collective", collective_name);
    visitor.on_attribute("group_id", group_id);
    visitor.on_attribute("block", block);
    strategy = from_string(strategy_name);
    collective = collective_from_string(collective_name);
    return true;
}

std::string ProjectionRole::to_string() const {
    std::ostringstream out;
    out << to_cstr(strategy) << " " << to_cstr(collective) << " group " << group_id << " by " << block;
    return out.str();
}

bool ProjectionRole::operator==(const ProjectionRole& other) const {
    return strategy == other.strategy && collective == other.collective && group_id == other.group_id;
}

void set_shard(const std::shared_ptr<ov::Node>& node, const ShardAnnotation& annotation) {
    set_annotation(node, annotation);
}

void set_role(const std::shared_ptr<ov::Node>& node, const ProjectionRole& role) {
    set_annotation(node, role);
}

const ShardAnnotation* get_shard(const ov::Node& node) {
    return get_annotation<ShardAnnotation>(node);
}

const ProjectionRole* get_role(const ov::Node& node) {
    return get_annotation<ProjectionRole>(node);
}

void strip_annotations(const ov::Model& model) {
    for (const auto& node : model.get_ops()) {
        auto& rt_info = node->get_rt_info();
        rt_info.erase(ShardAnnotation::get_type_info_static());
        rt_info.erase(ProjectionRole::get_type_info_static());
    }
}

std::vector<Range> rank_ranges(const std::vector<Segment>& segments, size_t rank, size_t world_size) {
    OPENVINO_ASSERT(world_size > 0 && rank < world_size, "[TP] rank ", rank, " of ", world_size);
    std::vector<Range> ranges;
    size_t segment_offset = 0;
    for (const auto& segment : segments) {
        OPENVINO_ASSERT(segment.unit > 0 && segment.length % segment.unit == 0,
                        "[TP] cannot split ",
                        segment.length,
                        " elements in units of ",
                        segment.unit);
        const size_t units = segment.length / segment.unit;
        const size_t shares = segment.count ? segment.count : world_size;
        OPENVINO_ASSERT(segment.step > 0 && shares * segment.step <= world_size && world_size % segment.step == 0,
                        "[TP] ", shares, " shares of ", segment.step, " ranks each do not fit ", world_size, " ranks");
        const size_t share = (rank / segment.step) % shares;
        const size_t base = units / shares;
        const size_t remainder = units % shares;
        const size_t offset = (share * base + std::min(share, remainder)) * segment.unit;
        const size_t length = (base + (share < remainder ? 1 : 0)) * segment.unit;
        ranges.push_back({segment_offset + offset, length});
        segment_offset += segment.length;
    }
    return ranges;
}

size_t rank_length(const std::vector<Segment>& segments, size_t rank, size_t world_size) {
    size_t total = 0;
    for (const auto& range : rank_ranges(segments, rank, world_size)) {
        total += range.length;
    }
    return total;
}

int64_t localize(const ShardAnnotation& annotation, size_t listed, int64_t global, size_t rank, size_t world_size) {
    const auto& segments = annotation.segments;
    switch (annotation.element_mode) {
    case ShardAnnotation::Elements::SIZE:
        return static_cast<int64_t>(rank_length(segments, rank, world_size));
    case ShardAnnotation::Elements::SEGMENT_SIZES:
        OPENVINO_ASSERT(listed < segments.size(), "[TP] element ", listed, " of ", segments.size(), " segments");
        // -1: the remainder, inferred on the rank as well.
        return global < 0 ? global : static_cast<int64_t>(rank_length({segments[listed]}, rank, world_size));
    case ShardAnnotation::Elements::BOUNDS: {
        int64_t boundary = 0;
        int64_t local = 0;
        for (size_t k = 0;; ++k) {
            if (global == boundary)
                return local;
            if (k == segments.size())
                break;
            boundary += static_cast<int64_t>(segments[k].length);
            local += static_cast<int64_t>(rank_length({segments[k]}, rank, world_size));
        }
        // Past the end (INT64_MAX as "to the end"): clamped on the rank as well.
        OPENVINO_ASSERT(global > boundary, "[TP] ", global, " is not a boundary between the segments of ", annotation.to_string());
        return global;
    }
    }
    OPENVINO_THROW("[TP] unknown element mode");
}

}  // namespace tp
}  // namespace ov
