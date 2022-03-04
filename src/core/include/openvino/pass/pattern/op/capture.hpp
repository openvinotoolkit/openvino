// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"

namespace ov {
namespace pass {
namespace pattern {
namespace op {
/// Experimental for support of recurrent matches.
///
/// Capture adds the pattern value map to a list of pattern value maps and resets
/// matches for pattern nodes not in the static node list. The match always succeeds.
class OPENVINO_API Capture : public Pattern {
public:
    OPENVINO_RTTI("patternCapture");
    BWDCMP_RTTI_DECLARATION;
    Capture(const Output<Node>& arg) : Pattern({arg}) {
        set_output_type(0, arg.get_element_type(), arg.get_partial_shape());
    }

    /// \brief static nodes are retained after a capture. All other nodes are dropped
    std::set<Node*> get_static_nodes() {
        return m_static_nodes;
    }
    void set_static_nodes(const std::set<Node*>& static_nodes) {
        m_static_nodes = static_nodes;
    }

    bool match_value(pattern::Matcher* matcher,
                     const Output<Node>& pattern_value,
                     const Output<Node>& graph_value) override;

protected:
    std::set<Node*> m_static_nodes;
};
}  // namespace op
}  // namespace pattern
}  // namespace pass
}  // namespace ov
