// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/op.hpp"
#include "snippets/op/tile_helpers.hpp"

namespace ngraph {
namespace snippets {
namespace op {
std::shared_ptr<TileBegin> insertTileBeginAfterOutputs(const OutputVector& originalOutputs, size_t dimension, size_t workAmount,
                                                       size_t increment, std::vector<bool> apply_increment,
                                                       std::vector<int64_t> finalization_offsets) {
    std::vector<std::set<Input<Node>>> originalChildInputs;
    for (const auto& out : originalOutputs) {
        originalChildInputs.push_back(out.get_target_inputs());
    }

    auto tileBegin = std::make_shared<TileBegin>(originalOutputs, dimension, workAmount, increment, std::move(apply_increment),
                                                 std::move(finalization_offsets));

    for (int i = 0; i < originalChildInputs.size(); i++) {
        for (auto& input : originalChildInputs[i]) {
            input.replace_source_output(tileBegin->output(i));
        }
    }
    return tileBegin;
}

std::shared_ptr<TileEnd> insertTileEndBeforeInputs(const std::vector<Input<Node>>& originalInputs,
                                                   const std::shared_ptr<TileBegin>& tileBegin) {
    OutputVector originalParentOutputs;
    for (const auto& in : originalInputs) {
        originalParentOutputs.push_back(in.get_source_output());
    }
    originalParentOutputs.push_back(tileBegin->output(tileBegin->get_output_size() - 1));
    auto tileEnd = std::make_shared<TileEnd>(originalParentOutputs);

    for (int i = 0; i < originalInputs.size(); i++) {
        originalInputs[i].replace_source_output(tileEnd->output(i));
    }
    return tileEnd;
}

} // namespace op
} // namespace snippets
} // namespace ngraph