// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

/// @brief Replaces remaining SequenceMark nodes with Concat operations.
/// This transformation should run after all other transformations that might
/// consume or remove SequenceMark nodes. It concatenates all inputs of the
/// SequenceMark into a single 1D tensor.
class SequenceMarkReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::SequenceMarkReplacer");
    SequenceMarkReplacer();
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
