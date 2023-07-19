// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_validation.hpp"
#include "static_shape.hpp"

namespace ov {
namespace intel_cpu {

namespace {
void partial_shape_convert_throw() {
    OPENVINO_THROW("[shape infer] Shouldn't convert from PartialShape to StaticShape at runtime.");
}
}  // namespace

template <class T>
bool merge_into(StaticShapeCon& dst, const T& src) {
    auto success = (dst.size() != src.size());

    for (size_t i = 0; success && (i < dst.size()); ++i)
        success = StaticDimension::merge(dst[i], dst[i], src[i]);

    return success;
}

StaticShape::StaticShape(std::vector<StaticDimension> dimensions)
        : std::vector<StaticDimension>(std::move(dimensions)) {}

StaticShape::StaticShape(const std::vector<StaticDimension::value_type>& dimensions)
        : std::vector<StaticDimension>(dimensions.begin(), dimensions.end()) {}

StaticShape::StaticShape(std::initializer_list<StaticDimension> init)
        : std::vector<StaticDimension>(init.begin(), init.end()) {}

StaticShape::StaticShape(const ov::PartialShape&) : std::vector<StaticDimension>{} {
    partial_shape_convert_throw();
}

ov::Shape StaticShape::get_max_shape() const {
    return (*this).to_shape();
}

ov::Shape StaticShape::get_min_shape() const {
    return (*this).to_shape();
}

ov::Shape StaticShape::get_shape() const {
    return (*this).to_shape();
}

StaticShape operator+(const StaticShape& s1, const StaticShape& s2) {
    if (s1.size() != s2.size()) {
        throw std::invalid_argument("rank mismatch");
    }

    std::vector<StaticDimension> result(s1.size());
    for (size_t i = 0; i < s1.size(); ++i)
        result[i] = (s1[i] + s2[i]);
    return result;
}

std::ostream& operator<<(std::ostream& str, const StaticShape& shape) {
    str << "{";
    bool first = true;
    for (const auto& d : shape) {
        if (!first) str << ",";
        str << d;
        first = false;
    }
    return (str << "}");
}

bool StaticShape::compatible(const StaticShape& s) const {
    if (size() != s.size())
        return false;
    for (size_t i = 0; i < size(); ++i)
        if (!((*this)[i]).compatible(s[i]))
            return false;
    return true;
}

bool StaticShape::same_scheme(const StaticShape& s) const {
    if (size() != s.size())
        return false;
    for (size_t i = 0; i < size(); ++i)
        if (!((*this)[i]).same_scheme(s[i]))
            return false;
    return true;
}

bool StaticShape::merge_rank(const Rank& r) {
    if (r.is_dynamic()) {
        return true;
    } else {
        return (static_cast<int64_t>(size()) == r.get_length());
    }
}

ov::Shape StaticShape::to_shape() const {
    std::vector<size_t> shape_dimensions(size());
    std::transform(begin(), end(), shape_dimensions.begin(), [](const StaticDimension& d) {
        return d.get_length();
    });
    return shape_dimensions;
}

ov::PartialShape StaticShape::to_partial_shape() const {
    ov::PartialShape shape_dimensions = PartialShape::dynamic(size());
    std::transform(begin(), end(), shape_dimensions.begin(), [](const StaticDimension& d) {
        return d.get_length();
    });
    return shape_dimensions;
}

bool StaticShape::merge_into(StaticShape& dst, const StaticShape& src) {
    if (dst.size() != src.size())
        return false;
    bool success = true;
    for (size_t i = 0; i < dst.size(); ++i)
        success &= StaticDimension::merge(dst[i], dst[i], src[i]);
    return success;
}

bool StaticShape::broadcast_merge_into(StaticShape& dst,
                                       const StaticShape& src,
                                       const ov::op::AutoBroadcastSpec& autob) {
    switch (autob.m_type) {
    case ov::op::AutoBroadcastType::NONE:
        return true;
    case ov::op::AutoBroadcastType::NUMPY: {
        auto dst_rank = dst.size();
        auto src_rank = src.size();
        auto new_rank = std::max(dst_rank, src_rank);
        std::vector<StaticDimension> dims(new_rank);
        bool success = true;
        for (size_t i = 0; i < new_rank; i++) {
            auto dsti = i < (new_rank - dst_rank) ? StaticDimension(1) : dst[i - (new_rank - dst_rank)];
            auto srci = i < (new_rank - src_rank) ? StaticDimension(1) : src[i - (new_rank - src_rank)];
            success &= StaticDimension::broadcast_merge(dims[i], dsti, srci);
        }
        dst = StaticShape(std::move(dims));
        return success;
        }
        case ov::op::AutoBroadcastType::PDPD: {
            // Ranks are both static.
            auto dst_rank = dst.rank().get_length();
            auto src_rank = src.rank().get_length();
            // source rank can't be bigger than destination rank according to
            // PDPD broadcast rule.

            int64_t axis = autob.m_axis;
            if (src_rank > dst_rank || axis < -1)
                return false;

            axis = (axis == -1) ? (dst_rank - src_rank) : axis;
            if (src_rank + axis > dst_rank)
                return false;

            bool success = true;
            for (int64_t i = 0; i < src_rank; ++i) {
                if (src[i].get_length() > dst[axis + i].get_length())
                    return false;

                success &= StaticDimension::broadcast_merge(dst[axis + i], dst[axis + i], src[i]);
            }

            return success;
        }
        default:
            OPENVINO_THROW("Unsupported auto broadcast type: ", autob.m_type);
    }
    return false;
}

bool StaticShape::operator==(const StaticShape& shape) const {
    if (size() != shape.size())
        return false;
    for (size_t i = 0; i < size(); ++i)
        if ((*this)[i] != shape[i])
            return false;
    return true;
}

bool StaticShape::operator!=(const StaticShape& partial_shape) const {
    return !(*this == partial_shape);
}

//-- Shape as container
StaticShapeCon::StaticShapeAdapter() : m_dims{} {}
StaticShapeCon::StaticShapeAdapter(const TDims& dims) : m_dims{dims} {}
StaticShapeCon::StaticShapeAdapter(TDims&& dims) noexcept : m_dims{std::move(dims)} {}
StaticShapeCon::StaticShapeAdapter(const StaticShapeCon& other) : StaticShapeAdapter(*other) {}
StaticShapeCon::StaticShapeAdapter(const ov::PartialShape&) : m_dims{} {
    partial_shape_convert_throw();
}

ov::Rank StaticShapeCon::rank() const {
    return {static_cast<typename ov::Rank::value_type>(size())};
}

ov::Shape StaticShapeCon::to_shape() const {
    return {m_dims};
}

ov::Shape StaticShapeCon::get_max_shape() const {
    return to_shape();
}

ov::Shape StaticShapeCon::get_min_shape() const {
    return to_shape();
}

ov::Shape StaticShapeCon::get_shape() const {
    return to_shape();
}

ov::PartialShape StaticShapeCon::to_partial_shape() const {
    auto shape = PartialShape::dynamic(size());
    std::transform(m_dims.cbegin(), m_dims.cend(), shape.begin(), ov::util::Cast<typename PartialShape::value_type>());
    return shape;
}

bool StaticShapeCon::merge_rank(const ov::Rank& r) {
    return r.is_dynamic() || (size() == static_cast<size_t>(r.get_length()));
}

bool StaticShapeCon::merge_into(StaticShapeAdapter& dst, const StaticShapeAdapter& src) {
    return ov::intel_cpu::merge_into(dst, src);
}

bool StaticShapeCon::broadcast_merge_into(StaticShapeCon& dst,
                                          const StaticShapeCon& src,
                                          const ov::op::AutoBroadcastSpec& autob) {
    // Copy from StaticShape when broadcast shape infer reviewed.
    return false;
}

//-- Shape as reference
StaticShapeRef::StaticShapeAdapter(const ov::PartialShape&) : m_dims{} {
    partial_shape_convert_throw();
}

ov::Rank StaticShapeRef::rank() const {
    return {static_cast<typename ov::Rank::value_type>(size())};
}

bool StaticShapeRef::merge_rank(const ov::Rank& r) {
    return r.is_dynamic() || (size() == static_cast<size_t>(r.get_length()));
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

}   // namespace intel_cpu

template <>
void NodeValidationFailure::create(const CheckLocInfo& check_loc_info,
                                   std::pair<const Node*, const std::vector<intel_cpu::StaticShape>*>&& ctx,
                                   const std::string& explanation) {
    throw ov::NodeValidationFailure(make_what(check_loc_info,
                                              node_validation_failure_loc_string(ctx.first),
                                              ov::op::validate::shape_infer_explanation_str(*ctx.second, explanation)));
}
}   // namespace ov
