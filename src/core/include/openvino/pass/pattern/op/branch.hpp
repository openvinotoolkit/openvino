// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"

namespace ov {
namespace pass {
namespace pattern {
namespace op {
/// A branch adds a loop to the pattern. The branch match is successful if the
/// destination node pattern matches the graph value. The destination node is a node in
/// the pattern graph that will not have been created some time after the Branch node is
/// created; use set_destination to add it.
///
/// The branch destination is not stored as a shared pointer to prevent reference
/// cycles. Thus the destination node must be referenced in some other way to prevent it
/// from being deleted.
class OPENVINO_API Branch : public Pattern {
public:
    OPENVINO_RTTI("patternBranch");
    /// \brief Creates a Branch pattern
    Branch() : Pattern(OutputVector{}) {
        set_output_type(0, element::f32, Shape{});
    }

    void set_destination(const Output<Node>& destination) {
        m_destination_node = destination.get_node();
        m_destination_index = destination.get_index();
    }

    Output<Node> get_destination() const {
        return m_destination_node == nullptr
                   ? Output<Node>()
                   : Output<Node>{m_destination_node->shared_from_this(), m_destination_index};
    }

    bool match_value(pattern::Matcher* matcher,
                     const Output<Node>& pattern_value,
                     const Output<Node>& graph_value) override;

protected:
    Node* m_destination_node{nullptr};
    size_t m_destination_index{0};
};
}  // namespace op
}  // namespace pattern
}  // namespace pass
}  // namespace ov
