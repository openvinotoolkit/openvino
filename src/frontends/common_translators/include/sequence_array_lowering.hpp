// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace pass {

/// \brief Lowers loop-carried "sequence-of-N" patterns into N parallel tensors.
///
/// This pass complements SequenceIfReplacer by handling the cases that
/// SequenceIfReplacer cannot resolve directly — typically a loop-carried
/// sequence seeded from an empty SequenceMark and built/passed through on the
/// first iteration via an If(first_iter) branch. The classic instance is a
/// past-key/value cache: an outer empty sequence flows into a Loop body whose
/// body uses an If to either build a fresh length-N SequenceMark (iter 0) or
/// pass through the loop-carried Parameter (iter > 0).
///
/// Lowering principle: when a sequence has statically discoverable length N
/// and uniform per-slot shape/dtype, and all index/position operands of the
/// SequenceAt / SequenceInsert / SequenceErase helpers in its def-use closure
/// are compile-time constants, the sequence is rewritten as N parallel
/// tensor edges. Helper ops become slot-level operations:
///   SequenceMark(t_0..t_{N-1}) -> the N tensors
///   SequenceAt(seq, k)         -> seq.slots[k]
///   SequenceInsert(seq, v, pos)-> slots[:pos] + [v] + slots[pos:]
///   SequenceErase(seq, pos)    -> slots with element pos removed
///   SequenceLength(seq)        -> Constant<i64>(N)
/// MultiSubGraphOp (Loop / If) inputs and outputs carrying a sequence are
/// replaced with N parallel tensor inputs/outputs.
///
/// Conservative by design: any candidate that fails a precondition is left
/// unmodified so the universal unconverted-ops report can flag it.
class SequenceArrayLowering : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::pass::SequenceArrayLowering");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace frontend
}  // namespace ov
