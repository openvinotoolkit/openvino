// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "static_shape.hpp"

#include "shape_validation.hpp"

namespace ov {
namespace intel_cpu {

namespace {
void partial_shape_convert_throw() {
    OPENVINO_THROW("[shape infer] Shouldn't convert from PartialShape to StaticShape at runtime.");
}
}  // namespace

template <class T>
bool merge_into(StaticShape& dst, const T& src) {
    auto success = (dst.size() == src.size());

    for (size_t i = 0; success && (i < dst.size()); ++i) {
        success = StaticDimension::merge(dst[i], dst[i], src[i]);
    }

    return success;
}

//-- Shape as container
StaticShape::StaticShapeAdapter() : m_dims{} {}
StaticShape::StaticShapeAdapter(const TDims& dims) : m_dims{dims} {}
StaticShape::StaticShapeAdapter(TDims&& dims) noexcept : m_dims{std::move(dims)} {}
StaticShape::StaticShapeAdapter(const StaticShape& other) : StaticShapeAdapter(*other) {}
StaticShape::StaticShapeAdapter(const ov::PartialShape&) : m_dims{} {
    partial_shape_convert_throw();
}

ov::Rank StaticShape::rank() const {
    return {static_cast<typename ov::Rank::value_type>(size())};
}

ov::Shape StaticShape::to_shape() const {
    return {m_dims};
}

ov::Shape StaticShape::get_max_shape() const {
    return to_shape();
}

ov::Shape StaticShape::get_min_shape() const {
    return to_shape();
}

ov::Shape StaticShape::get_shape() const {
    return to_shape();
}

ov::PartialShape StaticShape::to_partial_shape() const {
    auto shape = PartialShape::dynamic(size());
    std::transform(m_dims.cbegin(), m_dims.cend(), shape.begin(), ov::util::Cast<typename PartialShape::value_type>());
    return shape;
}

bool StaticShape::merge_rank(const ov::Rank& r) {
    return r.is_dynamic() || (size() == static_cast<size_t>(r.get_length()));
}

bool StaticShape::merge_into(StaticShapeAdapter& dst, const StaticShapeAdapter& src) {
    return ov::intel_cpu::merge_into(dst, src);
}

bool StaticShape::broadcast_merge_into(StaticShape& dst,
                                       const StaticShapeAdapter& src,
                                       const ov::op::AutoBroadcastSpec& autob) {
    // Copy from StaticShape when broadcast shape infer reviewed.
    switch (autob.m_type) {
    case ov::op::AutoBroadcastType::NONE:
        return true;
    case ov::op::AutoBroadcastType::NUMPY: {
        auto dst_rank = dst.size();
        auto src_rank = src.size();
        auto new_rank = std::max(dst_rank, src_rank);
        StaticShape merged;
        merged.resize(new_rank);

        bool success = true;
        for (size_t i = 0; i < new_rank; i++) {
            auto dsti = i < (new_rank - dst_rank) ? StaticDimension(1) : dst[i - (new_rank - dst_rank)];
            auto srci = i < (new_rank - src_rank) ? StaticDimension(1) : src[i - (new_rank - src_rank)];
            success &= StaticDimension::broadcast_merge(merged[i], dsti, srci);
        }
        *dst = std::move(*merged);
        return success;
    }
    case ov::op::AutoBroadcastType::PDPD: {
        // Ranks are both static.
        auto dst_rank = dst.rank().get_length();
        auto src_rank = src.rank().get_length();
        // source rank can't be bigger than destination rank according to
        // PDPD broadcast rule.

        int64_t axis = autob.m_axis;
        if (src_rank > dst_rank || axis < -1) {
            return false;
        }

        axis = (axis == -1) ? (dst_rank - src_rank) : axis;
        if (src_rank + axis > dst_rank) {
            return false;
        }

        bool success = true;
        for (int64_t i = 0; i < src_rank; ++i) {
            if (src[i].get_length() > dst[axis + i].get_length()) {
                return false;
            }

            success &= StaticDimension::broadcast_merge(dst[axis + i], dst[axis + i], src[i]);
        }

        return success;
    }
    default:
        OPENVINO_THROW("Unsupported auto broadcast type: ", autob.m_type);
    }
    return false;
}

//-- Shape as reference
StaticShapeRef::StaticShapeAdapter(const StaticShape& shape) : m_dims{&(*shape)} {}
StaticShapeRef::StaticShapeAdapter(const ov::PartialShape&) {
    partial_shape_convert_throw();
}

ov::Rank StaticShapeRef::rank() const {
    return {static_cast<typename ov::Rank::value_type>(size())};
}

bool StaticShapeRef::merge_rank(const ov::Rank& r) {
    return r.is_dynamic() || (size() == static_cast<size_t>(r.get_length()));
}

ov::PartialShape StaticShapeRef::to_partial_shape() const {
    return {to_shape()};
}

ov::Shape StaticShapeRef::to_shape() const {
    return m_dims ? ov::Shape{*m_dims} : ov::Shape{};
}

ov::Shape StaticShapeRef::get_max_shape() const {
    return to_shape();
}

ov::Shape StaticShapeRef::get_min_shape() const {
    return to_shape();
}

ov::Shape StaticShapeRef::get_shape() const {
    return to_shape();
}

}  // namespace intel_cpu

template <>
void NodeValidationFailure::create(const char* file,
                                   int line,
                                   const char* check_string,
                                   std::pair<const Node*, const std::vector<intel_cpu::StaticShape>*>&& ctx,
                                   const std::string& explanation) {
    throw ov::NodeValidationFailure(make_what(file,
                                              line,
                                              check_string,
                                              node_validation_failure_loc_string(ctx.first),
                                              ov::op::validate::shape_infer_explanation_str(*ctx.second, explanation)));
}

template <>
void NodeValidationFailure::create(const char* file,
                                   int line,
                                   const char* check_string,
                                   std::pair<const Node*, const std::vector<intel_cpu::StaticShapeRef>*>&& ctx,
                                   const std::string& explanation) {
    throw ov::NodeValidationFailure(make_what(file,
                                              line,
                                              check_string,
                                              node_validation_failure_loc_string(ctx.first),
                                              ov::op::validate::shape_infer_explanation_str(*ctx.second, explanation)));
}
}  // namespace ov
