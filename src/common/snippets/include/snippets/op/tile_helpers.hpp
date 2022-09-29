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

std::shared_ptr<TileBegin> insertTileBeginAfterOutputs(const OutputVector& originalOutputs, size_t dimension, size_t workAmount,
                                                       size_t increment, std::vector<bool> apply_increment = {},
                                                       std::vector<int64_t> finalization_offsets = {});

std::shared_ptr<TileEnd> insertTileEndBeforeInputs(const std::vector<Input<Node>>& originalInputs,
                                                  const std::shared_ptr<TileBegin>& tileBegin);
template<typename T, typename ...Args>
std::shared_ptr<TileBegin> insertTileBegin(const T& afterTheseNodes, Args ...args) {
    static_assert(std::is_same<T, ParameterVector>() || std::is_same<T, NodeVector>(),
                  "Unsupported template parameter for insertTileBegin. Only ParameterVector or NodeVector is allowed");
    OutputVector originalOutputs;
    std::vector<std::set<Input<Node>>> childInputs;
    for (const auto &n : afterTheseNodes) {
        const auto& nodeOutputs = n->outputs();
        // Ignore the TileBegin->TileEnd edge to make it easier to construct enclosed Tiles
        std::move(nodeOutputs.begin(), nodeOutputs.end() - 1 * ov::is_type<TileBegin>(n), std::back_inserter(originalOutputs));
    }

    return insertTileBeginAfterOutputs(originalOutputs, args...);
}

template<typename ...Args>
std::shared_ptr<TileBegin> insertTileBegin(const OutputVector& afterTheseNodes, Args ...args) {
   return insertTileBeginAfterOutputs(afterTheseNodes, args...);
}

template<typename T>
std::shared_ptr<TileEnd> insertTileEnd(const T& beforeTheseNodes, const std::shared_ptr<TileBegin>& tileBegin) {
    static_assert(std::is_same<T, ResultVector>() || std::is_same<T, NodeVector>(),
                  "Unsupported template parameter for insertTileBegin. Only ParameterVector or NodeVector is allowed");
    std::vector<Input<Node>> originalInputs;
    for (const auto &n : beforeTheseNodes) {
        const auto& nodeInputs = n->inputs();
        // Ignore the TileBegin->TileEnd edge to facilitate enclosed Tiles construction
        std::move(nodeInputs.begin(), nodeInputs.end() - 1 * ov::is_type<TileEnd>(n), std::back_inserter(originalInputs));
    }
    return insertTileEndBeforeInputs(originalInputs, tileBegin);
}
template<>
inline std::shared_ptr<TileEnd> insertTileEnd(const std::vector<Input<Node>>& beforeTheseNodes, const std::shared_ptr<TileBegin>& tileBegin) {
    return insertTileEndBeforeInputs(beforeTheseNodes, tileBegin);
}

} // namespace op
} // namespace snippets
} // namespace ngraph