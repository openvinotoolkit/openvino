// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/parameter.hpp"
#include "tile.hpp"

namespace ngraph {
namespace snippets {
namespace op {

std::shared_ptr<TileBegin> insertTileBeginAfterOutputs(const OutputVector& originalOutputs);

std::shared_ptr<TileEnd> insertTileEndBeforeInputs(const std::vector<Input<Node>>& originalInputs,
                                                  const std::shared_ptr<TileBegin>& tileBegin,
                                                  size_t dimension, size_t work_amount, size_t increment,
                                                  std::vector<bool> apply_increment = {},
                                                  std::vector<int64_t> finalization_offsets = {});
template<typename T>
std::shared_ptr<TileBegin> insertTileBegin(const T& afterTheseNodes) {
    static_assert(std::is_same<T, ParameterVector>() || std::is_same<T, NodeVector>(),
                  "Unsupported template parameter for insertTileBegin. Only ParameterVector or NodeVector is allowed");
    OutputVector originalOutputs;
    std::vector<std::set<Input<Node>>> childInputs;
    for (const auto &n : afterTheseNodes) {
        const auto& nodeOutputs = n->outputs();
        // Ignore the TileBegin->TileEnd edge to make it easier to construct enclosed Tiles
        std::move(nodeOutputs.begin(), nodeOutputs.end() - 1 * ov::is_type<TileBegin>(n), std::back_inserter(originalOutputs));
    }

    return insertTileBeginAfterOutputs(originalOutputs);
}

template<>
inline std::shared_ptr<TileBegin> insertTileBegin(const OutputVector& afterTheseNodes) {
   return insertTileBeginAfterOutputs(afterTheseNodes);
}

template<typename T, typename ...Args>
std::shared_ptr<TileEnd> insertTileEnd(const T& beforeTheseNodes, Args ...args) {
    static_assert(std::is_same<T, NodeVector>(),
                  "Unsupported template parameter for insertTileEnd. Only NodeVector is allowed");
    std::vector<Input<Node>> originalInputs;
    for (const auto &n : beforeTheseNodes) {
        const auto& nodeInputs = n->inputs();
        // Ignore the TileBegin->TileEnd edge to facilitate enclosed Tiles construction
        std::move(nodeInputs.begin(), nodeInputs.end() - 1 * ov::is_type<TileEnd>(n), std::back_inserter(originalInputs));
    }
    return insertTileEndBeforeInputs(originalInputs, args...);
}
template<typename ...Args>
std::shared_ptr<TileEnd> insertTileEnd(const std::vector<Input<Node>>& beforeTheseNodes,  Args ...args) {
    return insertTileEndBeforeInputs(beforeTheseNodes, args...);
}
template<typename ...Args>
std::shared_ptr<TileEnd> insertTileEnd(const ResultVector& beforeTheseNodes,  Args ...args) {
    // Note that topological sort parses node arguments in reversed order, but results are added  - in direct order
    // So ve need to pass the reversed results to TileEnd to keep the original traversal order in topological sorter
    ov::NodeVector reversedResults(beforeTheseNodes.rbegin(), beforeTheseNodes.rend());
    return insertTileEnd(reversedResults, args...);
}

} // namespace op
} // namespace snippets
} // namespace ngraph