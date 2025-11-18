// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/partial_shape.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

#include "openvino/core/dimension.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/util/common_util.hpp"

ov::PartialShape::PartialShape() : PartialShape(std::initializer_list<Dimension>{}) {}

ov::PartialShape::PartialShape(std::initializer_list<Dimension> init) : PartialShape(true, init) {}

ov::PartialShape::PartialShape(const std::vector<Dimension::value_type>& dimensions)
    : m_rank_is_static(true),
      m_dimensions(dimensions.begin(), dimensions.end()) {}

ov::PartialShape::PartialShape(const Shape& shape)
    : m_rank_is_static(true),
      m_shape_type(ShapeType::SHAPE_IS_STATIC),
      m_dimensions(shape.begin(), shape.end()) {}

ov::PartialShape::PartialShape(const std::string& value) {
    auto val = ov::util::trim(value);
    if (val[0] == '[' && val[val.size() - 1] == ']')
        val = val.substr(1, val.size() - 2);
    val = ov::util::trim(val);
    if (val == "...") {
        m_rank_is_static = false;
        m_dimensions = std::vector<Dimension>();
        return;
    }
    m_rank_is_static = true;
    std::stringstream ss(val);
    std::string field;
    while (getline(ss, field, ',')) {
        OPENVINO_ASSERT(!field.empty(), "Cannot get vector of dimensions! \"" + value + "\" is incorrect");
        m_dimensions.emplace_back(field);
    }
}

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
        shape.reserve(rank().get_length());
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
        shape.reserve(rank().get_length());
        for (auto dimension : m_dimensions) {
            shape.push_back(dimension.get_interval().get_min_val());
        }
        return shape;
    }
}

ov::Shape ov::PartialShape::get_shape() const {
    OPENVINO_ASSERT(rank().is_static(), "get_shape() must be called on a static shape");
    Shape shape;
    shape.reserve(rank().get_length());
    for (auto dimension : m_dimensions) {
        auto min_val = dimension.get_interval().get_min_val();
        auto max_val = dimension.get_interval().get_max_val();
        OPENVINO_ASSERT(min_val == max_val, "get_shape() must be called on a static shape");
        shape.push_back(min_val);
    }
    return shape;
}

ov::PartialShape ov::operator+(const PartialShape& s1, const PartialShape& s2) {
    if (s1.rank().is_dynamic() || s2.rank().is_dynamic()) {
        return PartialShape::dynamic();
    }

    if (!s1.rank().compatible(s2.rank())) {
        OPENVINO_THROW("rank mismatch");
    }

    PartialShape result;
    result.m_rank_is_static = true;
    result.m_dimensions.reserve(s1.m_dimensions.size());
    for (size_t i = 0; i < s1.m_dimensions.size(); i++) {
        result.m_dimensions.push_back(s1.m_dimensions[i] + s2.m_dimensions[i]);
    }
    return result;
}

std::ostream& ov::operator<<(std::ostream& str, const PartialShape& shape) {
    if (shape.m_rank_is_static) {
        str << "[";
        bool first = true;
        for (auto& d : shape.m_dimensions) {
            if (!first) {
                str << ",";
            }
            str << d;
            first = false;
        }
        return (str << "]");
    } else {
        return (str << "[...]");
    }
}

std::string ov::PartialShape::to_string() const {
    std::stringstream shape_str_stream;
    shape_str_stream << *this;
    return shape_str_stream.str();
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

        for (int64_t i = 0; i < rank().get_length(); i++) {
            if (!m_dimensions[i].same_scheme(s.m_dimensions[i]))
                return false;
        }

        return true;
    } else {
        return false;
    }
}

bool ov::PartialShape::relaxes(const PartialShape& s) const {
    if (rank().is_dynamic()) {
        return true;
    } else if (s.rank().is_static() && rank().get_length() == s.rank().get_length()) {
        for (int64_t i = 0; i < rank().get_length(); i++) {
            if (!m_dimensions[i].relaxes(s.m_dimensions[i]))
                return false;
        }

        return true;
    } else {
        return false;
    }
}

bool ov::PartialShape::refines(const PartialShape& s) const {
    if (s.rank().is_dynamic()) {
        return true;
    } else if (rank().is_static() && rank().get_length() == s.rank().get_length()) {
        for (int64_t i = 0; i < rank().get_length(); i++) {
            if (!m_dimensions[i].refines(s.m_dimensions[i]))
                return false;
        }

        return true;
    } else {
        return false;
    }
}

bool ov::PartialShape::merge_rank(const Rank& r) {
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
        OPENVINO_THROW("to_shape was called on a dynamic shape.");
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
                const auto& dsti = i < (new_rank - dst_rank) ? Dimension(1) : dst[i - (new_rank - dst_rank)];
                const auto& srci = i < (new_rank - src_rank) ? Dimension(1) : src[i - (new_rank - src_rank)];
                success &= Dimension::broadcast_merge(dims[i], dsti, srci);
            }
            dst = PartialShape(std::move(dims));
            return success;
        }
    }
    case op::AutoBroadcastType::PDPD: {
        if (dst.rank().is_dynamic() || src.rank().is_dynamic()) {
            dst = PartialShape::dynamic();
            return true;
        } else {
            // Ranks are both static.
            auto dst_rank = dst.rank().get_length();
            auto src_rank = src.rank().get_length();

            int64_t axis = autob.m_axis;
            if (src_rank > dst_rank || axis < -1)
                return false;

            axis = (axis == -1) ? (dst_rank - src_rank) : axis;

            if (src_rank + axis > dst_rank)
                return false;

            bool success = true;
            for (int64_t i = 0; i < src_rank; ++i) {
                if (dst[axis + i].is_static() && src[i].is_static()) {
                    if (src[i].get_length() > dst[axis + i].get_length())
                        return false;
                }

                success &= Dimension::broadcast_merge(dst[axis + i], dst[axis + i], src[i]);
            }

            return success;
        }
    }
    default:
        OPENVINO_THROW("Unsupported auto broadcast type: ", autob.m_type);
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

const ov::Dimension& ov::PartialShape::operator[](std::ptrdiff_t i) const {
    return m_dimensions[util::normalize_shape_index(i, m_dimensions.size())];
}

ov::Dimension& ov::PartialShape::operator[](std::ptrdiff_t i) {
    m_shape_type = ShapeType::SHAPE_IS_UPDATED;  // We can't guarantee that the shape remains static or dynamic.
    return m_dimensions[util::normalize_shape_index(i, m_dimensions.size())];
}

ov::AttributeAdapter<ov::PartialShape>::~AttributeAdapter() = default;
