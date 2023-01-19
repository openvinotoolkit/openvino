// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/op.hpp"
#include "snippets/pass/loop_helpers.hpp"

namespace ngraph {
namespace snippets {
namespace op {
std::shared_ptr<LoopBegin> insertLoopBeginAfterOutputs(const OutputVector& originalOutputs) {
    std::vector<std::set<Input<Node>>> originalChildInputs;
    for (const auto& out : originalOutputs) {
        originalChildInputs.push_back(out.get_target_inputs());
    }

    auto loop_begin = std::make_shared<LoopBegin>(originalOutputs);

    for (int i = 0; i < originalChildInputs.size(); i++) {
        for (auto& input : originalChildInputs[i]) {
            input.replace_source_output(loop_begin->output(i));
        }
    }
    return loop_begin;
}

std::shared_ptr<LoopEnd> insertLoopEndBeforeInputs(const std::vector<Input<Node>>& originalInputs,
                                                   const std::shared_ptr<LoopBegin>& loopBegin,
                                                   size_t work_amount, size_t increment,
                                                   std::vector<bool> apply_increment,
                                                   std::vector<int64_t> finalization_offsets) {
    OutputVector originalParentOutputs;
    for (const auto& in : originalInputs) {
        originalParentOutputs.push_back(in.get_source_output());
    }
    originalParentOutputs.push_back(loopBegin->output(loopBegin->get_output_size() - 1));
    auto loop_end = std::make_shared<LoopEnd>(originalParentOutputs, work_amount, increment,
                                             std::move(apply_increment), std::move(finalization_offsets));

    for (int i = 0; i < originalInputs.size(); i++) {
        originalInputs[i].replace_source_output(loop_end->output(i));
    }
    return loop_end;
}

} // namespace op
} // namespace snippets
} // namespace ngraph