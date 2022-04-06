// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/partial_shape.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

#include "dimension_tracker.hpp"
#include "ngraph/check.hpp"

ov::PartialShape::PartialShape() : PartialShape(std::initializer_list<Dimension>{}) {}

ov::PartialShape::PartialShape(std::initializer_list<Dimension> init) : PartialShape(true, init) {}

ov::PartialShape::PartialShape(const std::vector<Dimension::value_type>& dimensions)
    : m_rank_is_static(true),
      m_dimensions(dimensions.begin(), dimensions.end()) {}

ov::PartialShape::PartialShape(const Shape& shape)
    : m_rank_is_static(true),
      m_shape_type(ShapeType::SHAPE_IS_STATIC),
      m_dimensions(shape.begin(), shape.end()) {}

ov::PartialShape::PartialShape(bool rank_is_static, std::vector<Dimension> dimensions)
    : m_rank_is_static(rank_is_static),
      m_dimensions(std::move(dimensions)) {}

ov::PartialShape::PartialShape(std::vector<Dimension> dimensions)
    : m_rank_is_static(true),
      m_dimensions(std::move(dimensions)) {}

bool ov::PartialShape::is_static() const {
    ShapeType shape_type = m_shape_type;

    if (m_shape_type == ShapeType::SHAPE_IS_UNKNOWN || m_shape_type == ShapeType::SHAPE_IS_UPDATED) {
        shape_type = m_rank_is_static && std::all_of(m_dimensions.begin(),
                                                     m_dimensions.end(),
                                                     [](const Dimension& d) {
                                                         return d.is_static();
                                                     })
                         ? ShapeType::SHAPE_IS_STATIC
                         : ShapeType::SHAPE_IS_DYNAMIC;

        if (m_shape_type == ShapeType::SHAPE_IS_UNKNOWN)
            m_shape_type = shape_type;
    }

    return shape_type == ShapeType::SHAPE_IS_STATIC;
}

bool ov::PartialShape::operator==(const PartialShape& partial_shape) const {
    if (rank() != partial_shape.rank()) {
        return false;
    }
    if (rank().is_dynamic()) {
        return true;
    }
    for (auto i = 0; i < rank().get_length(); ++i) {
        if (m_dimensions[i] != partial_shape.m_dimensions[i]) {
            return false;
        }
    }
    return true;
}

bool ov::PartialShape::operator!=(const PartialShape& partial_shape) const {
    return !(*this == partial_shape);
}

ov::Shape ov::PartialShape::get_max_shape() const {
    if (rank().is_dynamic()) {
        return Shape();
    } else {
        Shape shape;
        for (auto dimension : m_dimensions) {
            shape.push_back(dimension.get_interval().get_max_val());
        }
        return shape;
    }
}

ov::Shape ov::PartialShape::get_min_shape() const {
    if (rank().is_dynamic()) {
        return Shape();
    } else {
        Shape shape;
        for (auto dimension : m_dimensions) {
            shape.push_back(dimension.get_interval().get_min_val());
        }
        return shape;
    }
}

ov::Shape ov::PartialShape::get_shape() const {
    NGRAPH_CHECK(rank().is_static(), "get_shape() must be called on a static shape");
    Shape shape;
    for (auto dimension : m_dimensions) {
        auto min_val = dimension.get_interval().get_min_val();
        auto max_val = dimension.get_interval().get_max_val();
        NGRAPH_CHECK(min_val == max_val, "get_shape() must be called on a static shape");
        shape.push_back(min_val);
    }
    return shape;
}

ov::PartialShape ov::operator+(const PartialShape& s1, const PartialShape& s2) {
    if (s1.rank().is_dynamic() || s2.rank().is_dynamic()) {
        return PartialShape::dynamic();
    }

    if (!s1.rank().compatible(s2.rank())) {
        throw std::invalid_argument("rank mismatch");
    }

    PartialShape result{};
    result.m_rank_is_static = true;
    for (size_t i = 0; i < s1.m_dimensions.size(); i++) {
        result.m_dimensions.push_back(s1.m_dimensions[i] + s2.m_dimensions[i]);
    }
    return result;
}

std::ostream& ov::operator<<(std::ostream& str, const PartialShape& shape) {
    if (shape.m_rank_is_static) {
        str << "{";
        bool first = true;
        for (auto& d : shape.m_dimensions) {
            if (!first) {
                str << ",";
            }
            if (const auto& l = ov::DimensionTracker::get_label(d))
                str << "l<" << l << ">";
            str << d;
            first = false;
        }
        return (str << "}");
    } else {
        return (str << "...");
    }
}

ov::PartialShape ov::PartialShape::dynamic(Rank r) {
    return PartialShape(r.is_static(),
                        std::vector<Dimension>(r.is_static() ? r.get_length() : 0, Dimension::dynamic()));
}

bool ov::PartialShape::compatible(const PartialShape& s) const {
    // If we don't know *this's rank, or we don't know s's rank, they are compatible.
    if (!m_rank_is_static || s.rank().is_dynamic()) {
        return true;
    }
    // If we do know *this's rank and s's rank, and they are unequal, they are incompatible.
    else if (rank().get_length() != s.rank().get_length()) {
        return false;
    }
    // If we know both the ranks and they are equal, then *this and s are compatible iff they
    // are elementwise compatible everywhere.
    else {
        for (int64_t i = 0; i < rank().get_length(); i++) {
            if (!m_dimensions[i].compatible(s.m_dimensions[i])) {
                return false;
            }
        }
        // If we are still here, we know that s1 and s2 have the same rank and are elementwise
        // compatible everywhere.
        return true;
    }
}

bool ov::PartialShape::same_scheme(const PartialShape& s) const {
    if (rank().is_dynamic() && s.rank().is_dynamic()) {
        return true;
    } else if (rank().is_static() && s.rank().is_static()) {
        if (rank().get_length() != s.rank().get_length()) {
            return false;
        }

        bool success = true;

        for (int64_t i = 0; i < rank().get_length(); i++) {
            success &= (*this)[i].same_scheme(s[i]);
        }

        return success;
    } else {
        return false;
    }
}

bool ov::PartialShape::relaxes(const PartialShape& s) const {
    if (rank().is_dynamic()) {
        return true;
    } else if (s.rank().is_static() && rank().get_length() == s.rank().get_length()) {
        bool all_relax = true;

        for (int64_t i = 0; i < rank().get_length(); i++) {
            all_relax &= ((*this)[i].relaxes(s[i]));
        }

        return all_relax;
    } else {
        return false;
    }
}

bool ov::PartialShape::refines(const PartialShape& s) const {
    if (s.rank().is_dynamic()) {
        return true;
    } else if (rank().is_static() && rank().get_length() == s.rank().get_length()) {
        bool all_refine = true;

        for (int64_t i = 0; i < rank().get_length(); i++) {
            all_refine &= ((*this)[i].refines(s[i]));
        }

        return all_refine;
    } else {
        return false;
    }
}

bool ov::PartialShape::merge_rank(Rank r) {
    if (r.is_dynamic()) {
        return true;
    } else if (!m_rank_is_static) {
        m_rank_is_static = true;
        m_dimensions = std::vector<Dimension>(r.get_length(), Dimension::dynamic());
        m_shape_type = ShapeType::SHAPE_IS_UNKNOWN;
        return true;
    } else {
        return (static_cast<int64_t>(m_dimensions.size()) == r.get_length());
    }
}

ov::Shape ov::PartialShape::to_shape() const {
    if (is_dynamic()) {
        throw std::invalid_argument("to_shape was called on a dynamic shape.");
    }

    std::vector<size_t> shape_dimensions(m_dimensions.size());
    std::transform(m_dimensions.begin(), m_dimensions.end(), shape_dimensions.begin(), [](const Dimension& d) {
        return d.get_length();
    });

    return shape_dimensions;
}

bool ov::PartialShape::merge_into(PartialShape& dst, const PartialShape& src) {
    if (dst.rank().is_dynamic()) {
        dst = src;
        return true;
    } else if (src.rank().is_dynamic()) {
        // No change to dst.
        return true;
    } else if (dst.rank().get_length() != src.rank().get_length()) {
        // Mismatching static ranks, cannot merge.
        return false;
    } else {
        // Ranks are both static, and they match.
        bool success = true;
        for (int64_t i = 0; i < dst.rank().get_length(); i++) {
            success &= Dimension::merge(dst[i], dst[i], src[i]);
        }
        return success;
    }
}

bool ov::PartialShape::broadcast_merge_into(PartialShape& dst,
                                            const PartialShape& src,
                                            const op::AutoBroadcastSpec& autob) {
    switch (autob.m_type) {
    case op::AutoBroadcastType::NONE:
        return true;
    case op::AutoBroadcastType::NUMPY: {
        if (dst.rank().is_dynamic() || src.rank().is_dynamic()) {
            dst = PartialShape::dynamic();
            return true;
        } else {
            // Ranks are both static.
            auto dst_rank = dst.rank().get_length();
            auto src_rank = src.rank().get_length();
            auto new_rank = std::max(dst_rank, src_rank);
            std::vector<Dimension> dims(new_rank);
            bool success = true;
            for (int64_t i = 0; i < new_rank; i++) {
                auto dsti = i < (new_rank - dst_rank) ? Dimension(1) : dst[i - (new_rank - dst_rank)];
                auto srci = i < (new_rank - src_rank) ? Dimension(1) : src[i - (new_rank - src_rank)];
                success &= Dimension::broadcast_merge(dims[i], dsti, srci);
            }
            dst = PartialShape(std::move(dims));
            return success;
        }
    }
    case op::AutoBroadcastType::PDPD: {
        if (dst.rank().is_dynamic() || src.rank().is_dynamic()) {
            return true;
        } else {
            // Ranks are both static.
            auto dst_rank = dst.rank().get_length();
            auto src_rank = src.rank().get_length();
            // source rank can't be bigger than destination rank according to PDPD broadcast rule.
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
    }
    default:
        NGRAPH_CHECK(false, "Unsupported auto broadcast type: ", autob.m_type);
    }

    return false;
}

bool ov::PartialShape::all_non_negative() const {
    for (auto& d : m_dimensions) {
        if (d.is_static() && d.get_length() < 0) {
            return false;
        }
    }

    return true;
}

const ov::Dimension& ov::PartialShape::operator[](size_t i) const {
    if (i >= m_dimensions.size()) {
        throw std::out_of_range("Accessing out-of-range dimension in Dimension[]");
    }
    return m_dimensions[i];
}

ov::Dimension& ov::PartialShape::operator[](size_t i) {
    if (i >= m_dimensions.size()) {
        throw std::out_of_range("Accessing out-of-range dimension in Dimension[]");
    }
    m_shape_type = ShapeType::SHAPE_IS_UPDATED;  // We can't guarantee that the shape remains static or dynamic.
    return m_dimensions[i];
}

BWDCMP_RTTI_DEFINITION(ov::AttributeAdapter<ov::PartialShape>);
