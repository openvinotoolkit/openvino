// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/rank.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/static_dimension.hpp"

namespace ngraph {
namespace op {
struct AutoBroadcastSpec;
}

/// \brief Class representing a shape that may be partially or totally dynamic.
///
/// XXX: THIS CLASS IS EXPERIMENTAL AND THE ENTIRE DESIGN IS SUBJECT TO CHANGE.
///
/// A PartialShape may have:
///
/// \li Dynamic rank. (Informal notation: `?`)
/// \li Static rank, but dynamic dimensions on some or all axes.
///     (Informal notation examples: `{1,2,?,4}`, `{?,?,?}`)
/// \li Static rank, and static dimensions on all axes.
///     (Informal notation examples: `{1,2,3,4}`, `{6}`, `{}`)
class NGRAPH_API StaticShape : public std::vector<StaticDimension> {
public:
    /// \brief Constructs a static StaticShape from a Shape.
    /// \param shape The Shape to convert into StaticShape.
    StaticShape() = default;
    StaticShape(const Shape& shape) : std::vector<StaticDimension>(shape.begin(), shape.end()) {}
    StaticShape(const PartialShape& shape) : std::vector<StaticDimension>(shape.begin(), shape.end()) {}

    static bool is_static() {
        return true;
    };
    static bool is_dynamic() {
        return false;
    }
    static bool all_non_negative() {
        return true;
    };

    Rank rank() const {
        return Rank(size());
    }
    bool merge_rank(Rank r);

    bool compatible(const StaticShape& s) const;
    bool same_scheme(const StaticShape& s) const {
        return compatible(s);
    };
    bool relaxes(const StaticShape& s) const {
        return compatible(s);
    };
    bool refines(const StaticShape& s) const {
        return compatible(s);
    };

    Shape to_shape() const;
    Shape get_max_shape() const {
        return to_shape();
    };
    Shape get_min_shape() const {
        return to_shape();
    };
    Shape get_shape() const {
        return to_shape();
    };

    static bool merge_into(StaticShape& dst, const StaticShape& src);
    static bool broadcast_merge_into(StaticShape& dst, const StaticShape& src, const op::AutoBroadcastSpec& autob);

    friend NGRAPH_API std::ostream& operator<<(std::ostream& str, const StaticShape& shape);
};

NGRAPH_API
std::ostream& operator<<(std::ostream& str, const StaticShape& shape);

}  // namespace ngraph
