// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "snippets/emitter.hpp"
#include "tile.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface TileScheduler
 * @brief Contains a set of Tiles (usually one vector and one scalar) and performs necessary preparations
 * before the Tiles could be executed: calculates offsets, sets proper work amounts, decrement pointers if the same data
 * have to read several times (broadcasting).
 * @ingroup snippets
 */
class TileScheduler : public ngraph::op::Op {
public:
    OPENVINO_OP("TileScheduler", "SnippetsOpset");

    TileScheduler(const std::pair<std::shared_ptr<ngraph::snippets::Emitter>, ngraph::snippets::RegInfo> &vector_region,
                  const std::pair<std::shared_ptr<ngraph::snippets::Emitter>, ngraph::snippets::RegInfo> &scalar_region);
    TileScheduler() = default;
    std::pair<std::shared_ptr<ngraph::snippets::Emitter>, ngraph::snippets::RegInfo> vector_region;
    std::pair<std::shared_ptr<ngraph::snippets::Emitter>, ngraph::snippets::RegInfo> scalar_region;
    // todo: this clone_with_new_inputs is irrelevant
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return std::make_shared<TileScheduler>(vector_region, scalar_region);
    }
//    std::shared_ptr<ngraph::snippets::op::Tile> vectorTile;
//    std::shared_ptr<ngraph::snippets::op::Tile> scalarTile;
    const void *compile_params;
};

} // namespace op
} // namespace snippets
} // namespace ngraph