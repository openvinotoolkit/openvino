// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace pass {

/// \brief Resolves SequenceAt / SequenceLength / SequenceErase helper operations.
///
/// Handles two patterns:
///   1. Helper applied directly to a SequenceMark / SequenceInsert chain - resolved
///      by indexing / counting / removing from the chain.
///   2. Helper applied to an output of an ov::op::v8::If whose corresponding Result
///      in both branches feeds from a SequenceMark / SequenceInsert chain - the
///      helper is pushed into the If by adding new tensor-typed outputs.
///
/// Iterates to fixpoint to handle nested If patterns.
class SequenceIfReplacer : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::pass::SequenceIfReplacer");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace frontend
}  // namespace ov
