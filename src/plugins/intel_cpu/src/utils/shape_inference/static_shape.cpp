// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "static_shape.hpp"

namespace ov {
namespace intel_cpu {

StaticShape::StaticShape(std::vector<StaticDimension> dimensions)
        : std::vector<StaticDimension>(std::move(dimensions)) {}

StaticShape::StaticShape(const std::vector<StaticDimension::value_type>& dimensions)
        : std::vector<StaticDimension>(dimensions.begin(), dimensions.end()) {}

StaticShape::StaticShape(std::initializer_list<StaticDimension> init)
        : std::vector<StaticDimension>(init.begin(), init.end()) {}


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

bool StaticShape::merge_rank(Rank r) {
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
                                       const ngraph::op::AutoBroadcastSpec& autob) {
    switch (autob.m_type) {
        case ngraph::op::AutoBroadcastType::NONE:
            return true;
        case ngraph::op::AutoBroadcastType::NUMPY: {
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
        case ngraph::op::AutoBroadcastType::PDPD: {
            // Ranks are both static.
            auto dst_rank = dst.rank().get_length();
            auto src_rank = src.rank().get_length();
            // source rank can't be bigger than destination rank according to
            // PDPD broadcast rule.
            if (src_rank > dst_rank)
              return false;
            if (dst_rank == src_rank && dst.compatible(src))
                return true;

            int64_t axis = autob.m_axis;
            if (axis < -1) {
                return false;
            }
            if (axis == -1) {
                axis = dst_rank - src_rank;
            }

            size_t len = src_rank;
            while (len > 0 && src[len - 1].is_static() && src[len - 1].get_length() == 1) {
                --len;
            }

            for (size_t i = axis; i < axis + len; ++i) {
                if (!(dst[i].compatible(src[i - axis]))) {
                    return false;
                }
            }

            return true;
        }
        default:
            NGRAPH_CHECK(false, "Unsupported auto broadcast type: ", autob.m_type);
    }
    return false;
}

bool StaticShape::operator==(const StaticShape& shape) const {
    if (size() != shape.size())
        return false;
    for (auto i = 0; i < size(); ++i)
        if ((*this)[i] != shape[i])
            return false;
    return true;
}

bool StaticShape::operator!=(const StaticShape& partial_shape) const {
    return !(*this == partial_shape);
}

}   // namespace intel_cpu
}   // namespace ov
