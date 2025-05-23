// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"

namespace ov::pass::pattern::op {
/// \brief The match always succeeds.
class OPENVINO_API True : public Pattern {
public:
    OPENVINO_RTTI("patternTrue");
    /// \brief Always matches, does not add node to match list.
    True() : Pattern() {}
    bool match_value(pattern::Matcher* matcher,
                     const Output<Node>& pattern_value,
                     const Output<Node>& graph_value) override;
};
}  // namespace ov::pass::pattern::op
