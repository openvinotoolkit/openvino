// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {

/**
 * @interface SnippetsMarkSkippedBase
 * @brief Base class to mark operations that should be ignored by snippets on tokenization stage.
 */
class SnippetsMarkSkippedBase : public ov::pass::ModelPass {
protected:
    bool canBePerformedAsScaleShift(const std::shared_ptr<const Node> &node, const int channelAxis);
};

}   // namespace intel_cpu
}   // namespace ov
