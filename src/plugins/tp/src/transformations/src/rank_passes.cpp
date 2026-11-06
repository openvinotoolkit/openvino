// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tp_transformations/rank_passes.hpp"

#include <cstring>

#include "itt.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/slice.hpp"
#include "ov_ops/tp_collective.hpp"
#include "tp_transformations/annotations.hpp"

namespace ov {
namespace tp {

namespace {

/// The rank's ranges of `c` along `axis`, concatenated. One contiguous range is a view into the
/// original buffer; anything else is copied.
std::shared_ptr<ov::op::v0::Constant> slice_constant(const std::shared_ptr<ov::op::v0::Constant>& c,
                                                     int64_t axis,
                                                     const std::vector<Range>& ranges,
                                                     TPShardingStats& stats) {
    const auto& shape = c->get_shape();
    const auto et = c->get_element_type();
    OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int64_t>(shape.size()),
                    "[TP] '", c->get_friendly_name(), "': axis ", axis, " out of range for ", shape);

    size_t outer = 1;
    for (int64_t d = 0; d < axis; ++d)
        outer *= shape[d];
    size_t inner = 1;
    for (size_t d = static_cast<size_t>(axis) + 1; d < shape.size(); ++d)
        inner *= shape[d];

    const size_t bitwidth = et.bitwidth();
    auto elems_to_bytes = [&](size_t elems) -> size_t {
        if (bitwidth >= 8)
            return elems * (bitwidth / 8);
        const size_t per_byte = 8 / bitwidth;
        OPENVINO_ASSERT(elems % per_byte == 0, "[TP] Element count ", elems, " not byte-aligned for ", et);
        return elems / per_byte;
    };

    ov::Shape out_shape = shape;
    out_shape[axis] = 0;
    for (const auto& range : ranges) {
        OPENVINO_ASSERT(range.offset + range.length <= shape[axis],
                        "[TP] '", c->get_friendly_name(), "': range [", range.offset, ", ",
                        range.offset + range.length, ") out of bounds for ", shape[axis]);
        out_shape[axis] += range.length;
    }

    std::shared_ptr<ov::op::v0::Constant> sliced;
    const bool view = outer == 1 && ranges.size() == 1;
    if (view) {
        const auto* data = static_cast<const char*>(c->get_data_ptr()) + elems_to_bytes(ranges[0].offset * inner);
        sliced = std::make_shared<ov::op::v0::Constant>(et, out_shape, static_cast<const void*>(data), std::shared_ptr<void>(c));
    } else {
        const size_t row_bytes = elems_to_bytes(shape[axis] * inner);
        const size_t out_row_bytes = elems_to_bytes(out_shape[axis] * inner);
        std::vector<uint8_t> buffer(outer * out_row_bytes);
        const auto* src = static_cast<const uint8_t*>(c->get_data_ptr());
        for (size_t o = 0; o < outer; ++o) {
            size_t written = 0;
            for (const auto& range : ranges) {
                const size_t bytes = elems_to_bytes(range.length * inner);
                std::memcpy(buffer.data() + o * out_row_bytes + written,
                            src + o * row_bytes + elems_to_bytes(range.offset * inner),
                            bytes);
                written += bytes;
            }
        }
        sliced = std::make_shared<ov::op::v0::Constant>(et, out_shape, buffer.data());
        ++stats.copies;
        stats.copied_bytes += buffer.size();
    }
    if (view)
        ++stats.views;

    sliced->get_rt_info() = c->get_rt_info();  // decompression markers
    // The weightless cache reads a constant back from the model's .bin, where only a contiguous
    // slice of an unconverted constant is found as it is.
    auto& rt_info = sliced->get_rt_info();
    if (const auto it = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static()); it != rt_info.end()) {
        const auto attr = it->second.as<ov::WeightlessCacheAttribute>();
        if (view && attr.original_dtype == et && attr.original_size == c->get_byte_size()) {
            it->second = ov::WeightlessCacheAttribute(elems_to_bytes(ranges[0].length * inner),
                                                      attr.bin_offset + elems_to_bytes(ranges[0].offset * inner),
                                                      et);
        } else {
            rt_info.erase(it);
        }
    }
    sliced->set_friendly_name(c->get_friendly_name() + "_shard");
    return sliced;
}

/// A Slice (or a Concat of Slices) of `source` to the rank's ranges along `axis`.
ov::Output<ov::Node> runtime_slice(const ov::Output<ov::Node>& source, int64_t axis, const std::vector<Range>& ranges) {
    ov::OutputVector parts;
    for (const auto& range : ranges) {
        auto start = ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(range.offset)});
        auto stop = ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(range.offset + range.length)});
        auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
        parts.push_back(std::make_shared<ov::op::v8::Slice>(source, start, stop, step, axes)->output(0));
    }
    return parts.size() == 1 ? parts.front() : std::make_shared<ov::op::v0::Concat>(parts, axis)->output(0);
}

}  // namespace

bool TPWeightShardingPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    OV_ITT_SCOPED_TASK_BASE(itt::domains::tp_transforms, "TPWeightShardingPass");
    TPShardingStats stats;
    for (const auto& node : model->get_ordered_ops()) {
        const auto* annotation = get_shard(*node);
        if (!annotation)
            continue;
        const auto local = static_cast<int64_t>(rank_length(annotation->segments, m_rank, m_world_size));

        if (annotation->input >= 0) {
            auto input = node->input(static_cast<size_t>(annotation->input));
            input.replace_source_output(runtime_slice(input.get_source_output(),
                                                      annotation->axis,
                                                      rank_ranges(annotation->segments, m_rank, m_world_size)));
            ++stats.runtime_slices;
        }

        for (const auto& key : annotation->rt_keys) {
            auto& rt_info = node->get_rt_info();
            const auto it = rt_info.find(key);
            OPENVINO_ASSERT(it != rt_info.end(), "[TP] '", node->get_friendly_name(), "' has no rt_info '", key, "'");
            it->second = static_cast<size_t>(local);
        }

        if (auto read = ov::as_type_ptr<ov::op::v6::ReadValue>(node)) {
            auto info = read->get_variable()->get_info();
            info.data_shape[annotation->axis] = local;
            read->get_variable()->update(info);
            continue;
        }

        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
        if (!constant || annotation->input >= 0)
            continue;
        std::shared_ptr<ov::op::v0::Constant> replacement;
        if (!annotation->elements.empty()) {
            auto data = constant->cast_vector<int64_t>();
            for (size_t i = 0; i < annotation->elements.size(); ++i) {
                auto& value = data.at(annotation->elements[i]);
                value = localize(*annotation, i, value, m_rank, m_world_size);
            }
            replacement = ov::op::v0::Constant::create(constant->get_element_type(), constant->get_shape(), data);
        } else {
            replacement = slice_constant(constant,
                                         annotation->axis,
                                         rank_ranges(annotation->segments, m_rank, m_world_size),
                                         stats);
        }
        replacement->get_rt_info().erase(ShardAnnotation::get_type_info_static());
        constant->output(0).replace(replacement->output(0));
    }
    model->validate_nodes_and_infer_types();
    if (m_stats)
        *m_stats = stats;
    return true;
}

bool TPInsertCollectives::run_on_model(const std::shared_ptr<ov::Model>& model) {
    OV_ITT_SCOPED_TASK_BASE(itt::domains::tp_transforms, "TPInsertCollectives");
    for (const auto& node : model->get_ordered_ops()) {
        const auto* role = get_role(*node);
        if (!role || role->collective == Collective::NONE)
            continue;

        std::shared_ptr<ov::Node> collective;
        if (role->collective == Collective::ALL_REDUCE) {
            collective = std::make_shared<ov::op::internal::TPAllReduce>(node->output(0),
                                                                         static_cast<int>(m_rank),
                                                                         static_cast<int>(m_world_size),
                                                                         role->group_id,
                                                                         ov::op::internal::ReduceOp::SUM);
            collective->set_friendly_name("tp_allreduce/" + node->get_friendly_name());
        } else {
            collective = std::make_shared<ov::op::internal::TPGather>(node->output(0),
                                                                      static_cast<int>(m_rank),
                                                                      static_cast<int>(m_world_size),
                                                                      role->group_id,
                                                                      /*axis=*/-1);
            collective->set_friendly_name(node->get_friendly_name() + "/tp_gather");
        }
        for (const auto& target : node->output(0).get_target_inputs()) {
            if (target.get_node() != collective.get())
                target.replace_source_output(collective->output(0));
        }
    }
    model->validate_nodes_and_infer_types();
    return true;
}

}  // namespace tp
}  // namespace ov
