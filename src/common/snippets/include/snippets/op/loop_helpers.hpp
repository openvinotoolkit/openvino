// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/parameter.hpp"
#include "loop.hpp"

namespace ngraph {
namespace snippets {
namespace op {

std::shared_ptr<LoopBegin> insertLoopBeginAfterOutputs(const OutputVector& originalOutputs);

std::shared_ptr<LoopEnd> insertLoopEndBeforeInputs(const std::vector<Input<Node>>& originalInputs,
                                                  const std::shared_ptr<LoopBegin>& tileBegin,
                                                  size_t dimension, size_t work_amount, size_t increment,
                                                  std::vector<bool> apply_increment = {},
                                                  std::vector<int64_t> finalization_offsets = {});
template<typename T>
std::shared_ptr<LoopBegin> insertLoopBegin(const T& afterTheseNodes) {
    static_assert(std::is_same<T, ParameterVector>() || std::is_same<T, NodeVector>(),
                  "Unsupported template parameter for insertLoopBegin. Only ParameterVector or NodeVector is allowed");
    OutputVector originalOutputs;
    std::vector<std::set<Input<Node>>> childInputs;
    for (const auto &n : afterTheseNodes) {
        const auto& nodeOutputs = n->outputs();
        // Ignore the LoopBegin->LoopEnd edge to make it easier to construct enclosed Loops
        std::move(nodeOutputs.begin(), nodeOutputs.end() - 1 * ov::is_type<LoopBegin>(n), std::back_inserter(originalOutputs));
    }

    return insertLoopBeginAfterOutputs(originalOutputs);
}

template<>
inline std::shared_ptr<LoopBegin> insertLoopBegin(const OutputVector& afterTheseNodes) {
   return insertLoopBeginAfterOutputs(afterTheseNodes);
}

template<typename T, typename ...Args>
std::shared_ptr<LoopEnd> insertLoopEnd(const T& beforeTheseNodes, Args ...args) {
    static_assert(std::is_same<T, ResultVector>() || std::is_same<T, NodeVector>(),
                  "Unsupported template parameter for insertLoopBegin. Only ParameterVector or NodeVector is allowed");
    std::vector<Input<Node>> originalInputs;
    for (const auto &n : beforeTheseNodes) {
        const auto& nodeInputs = n->inputs();
        // Ignore the LoopBegin->LoopEnd edge to facilitate enclosed Loops construction
        std::move(nodeInputs.begin(), nodeInputs.end() - 1 * ov::is_type<LoopEnd>(n), std::back_inserter(originalInputs));
    }
    return insertLoopEndBeforeInputs(originalInputs, args...);
}
template<typename ...Args>
std::shared_ptr<LoopEnd> insertLoopEnd(const std::vector<Input<Node>>& beforeTheseNodes,  Args ...args) {
    return insertLoopEndBeforeInputs(beforeTheseNodes, args...);
}

} // namespace op
} // namespace snippets
} // namespace ngraph