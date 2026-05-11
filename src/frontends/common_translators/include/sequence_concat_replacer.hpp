// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace frontend {
namespace pass {

/// \brief Replaces ConcatFromSequence operations with standard OpenVINO ops.
///
/// This transformation handles three patterns:
///   1. SequenceMark -> ConcatFromSequence: Replaced with Concat (with optional Unsqueeze for new_axis mode).
///      Covers SequenceConstruct and PyTorch aten::append chains (each SequenceInsert wrapped in SequenceMark).
///   2. SequenceInsert chain -> ConcatFromSequence: Flattened to Concat (ONNX native pattern).
///      Pattern: SequenceEmpty → SequenceInsert(seq,t1) → SequenceInsert(seq,t2) → ConcatFromSequence.
///      Only append-only chains (no-position or position==-1) are handled statically.
///   3. Loop(SequenceInsert) -> ConcatFromSequence: The Loop is rewritten to use ConcatOutputDescription.
///
/// Used by both ONNX and PyTorch frontends.
class SequenceConcatReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pass::SequenceConcatReplacer");
    SequenceConcatReplacer();
};

}  // namespace pass
}  // namespace frontend
}  // namespace ov
