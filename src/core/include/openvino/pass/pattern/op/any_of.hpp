// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"

namespace ov::pass::pattern::op {
/// The graph value is added to the matched values list. If the predicate is true for
/// the
/// graph node, a submatch is performed on the input of AnyOf and each input of the
/// graph node. The first match that succeeds results in a successful match. Otherwise
/// the match fails.
///
/// AnyOf may be given a type and shape for use in strict mode.
class OPENVINO_API AnyOf : public Pattern {
public:
    OPENVINO_RTTI("patternAnyOf");
    /// \brief creates a AnyOf node containing a sub-pattern described by \sa type and
    ///        \sa shape.
    template <typename TPredicate>
    AnyOf(const element::Type& type, const PartialShape& s, const TPredicate& pred, const OutputVector& wrapped_values)
        : Pattern(wrapped_values, Predicate(pred)) {
        if (wrapped_values.size() != 1) {
            OPENVINO_THROW("AnyOf expects exactly one argument");
        }
        set_output_type(0, type, s);
    }
    template <typename TPredicate>
    AnyOf(const element::Type& type, const PartialShape& s, const TPredicate& pred, const NodeVector& wrapped_values)
        : AnyOf(type, s, Predicate(pred), as_output_vector(wrapped_values)) {}

    /// \brief creates a AnyOf node containing a sub-pattern described by the type and
    ///        shape of \sa node.
    template <typename TPredicate>
    AnyOf(const Output<Node>& node, const TPredicate& pred, const OutputVector& wrapped_values)
        : AnyOf(node.get_element_type(), node.get_partial_shape(), pred, wrapped_values) {}
    template <typename TPredicate>
    AnyOf(const std::shared_ptr<Node>& node, const TPredicate& pred, const NodeVector& wrapped_values)
        : AnyOf(node, Predicate(pred), as_output_vector(wrapped_values)) {}
    bool match_value(Matcher* matcher, const Output<Node>& pattern_value, const Output<Node>& graph_value) override;
};
}  // namespace ov::pass::pattern::op
