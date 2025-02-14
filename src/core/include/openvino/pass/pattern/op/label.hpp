// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"

namespace ov::pass::pattern {
namespace op {
/// Fails if the predicate returns false on the graph value.
///
/// The graph value is added to the matched values list. If the Label is already
/// associated with a value, the match succeeds if the value is the same as the graph
/// value. Otherwise, the label is associated with the graph value and the match
/// succeeds if the pattern input matches the graph value.
///
/// DEPRECATED: If no inputs are given to Label, a True node is serves as the input. If
/// more than one inputs are given, an Or pattern of the inputs serves as the input.
class OPENVINO_API Label : public Pattern {
public:
    OPENVINO_RTTI("patternLabel");
    /// \brief creates a Label node containing a sub-pattern described by \sa type and
    ///        \sa shape.
    ///
    /// this Label node can be bound only to the nodes in the input graph
    /// that match the pattern specified by \sa wrapped_nodes
    /// Example:
    /// \code{.cpp}
    /// auto add = a + b; // a and b are op::Parameter in this example
    /// auto label = std::make_shared<pattern::op::Label>(element::f32,
    ///                                                   PartialShape{2,2},
    ///                                                   nullptr,
    ///                                                   OutputVector{add});
    /// \endcode
    template <typename Fn, typename std::enable_if_t<std::is_constructible_v<Predicate, Fn>>* = nullptr>
    Label(const element::Type& type, const PartialShape& s, Fn pred, const OutputVector& wrapped_values = {})
        : Pattern(OutputVector{wrap_values(wrapped_values)}, Predicate(pred)) {
        set_output_type(0, type, s);
    }
    Label(const element::Type& type, const PartialShape& s, NodePredicate pred, const NodeVector& wrapped_values = {})
        : Label(type, s, as_value_predicate(pred), as_output_vector(wrapped_values)) {}

    /// \brief creates a Label node containing a sub-pattern described by the type and
    ///        shape of \sa node.
    ///
    /// this Label node can be bound only to the nodes in the input graph
    /// that match the pattern specified by \sa wrapped_values
    /// Example:
    /// \code{.cpp}
    /// auto add = a + b; // a and b are op::Parameter in this example
    /// auto label = std::make_shared<pattern::op::Label>(add,
    ///                                                   nullptr,
    ///                                                   OutputVector{add});
    /// \endcode
    template <typename Fn, typename std::enable_if_t<std::is_constructible_v<Predicate, Fn>>* = nullptr>
    Label(const Output<Node>& value, const Fn& pred, const OutputVector& wrapped_values = {})
        : Label(value.get_element_type(), value.get_partial_shape(), Predicate(pred), wrapped_values) {}
    Label(const Output<Node>& node, const NodePredicate& pred, const NodeVector& wrapped_values = {})
        : Label(node.get_element_type(),
                node.get_partial_shape(),
                as_value_predicate(pred),
                as_output_vector(wrapped_values)) {}

    explicit Label(const element::Type& type = element::dynamic, const PartialShape& s = PartialShape::dynamic())
        : Label(type, s, nullptr, OutputVector{}) {}
    explicit Label(const Output<Node>& value)
        : Label(value.get_element_type(), value.get_partial_shape(), nullptr, OutputVector{}) {}

    bool match_value(Matcher* matcher, const Output<Node>& pattern_value, const Output<Node>& graph_value) override;

protected:
    static Output<Node> wrap_values(const OutputVector& wrapped_values);
};
}  // namespace op

OPENVINO_API std::shared_ptr<Node> any_input();

template <typename Fn, typename std::enable_if_t<std::is_constructible_v<op::Predicate, Fn>>* = nullptr>
std::shared_ptr<Node> any_input(const Fn& pred) {
    return std::make_shared<pattern::op::Label>(element::dynamic, PartialShape::dynamic(), op::Predicate(pred));
}
}  // namespace ov::pass::pattern
