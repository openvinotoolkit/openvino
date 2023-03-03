// Copyright (C) 2022-2023 Intel Corporation
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
 * @brief Contains a set of Tiles (currently one vector and one scalar) and performs necessary preparations
 * before the Tiles could be executed: calculates offsets, sets proper work amounts, decrement pointers if the same data
 * have to be read several times (broadcasting).
 * @ingroup snippets
 */
class TileScheduler : public ngraph::op::Op {
public:
    OPENVINO_OP("TileScheduler", "SnippetsOpset");

    TileScheduler(const AllocatedEmitter& vector_region, const AllocatedEmitter& scalar_region);
    TileScheduler() = default;
    AllocatedEmitter vector_region;
    AllocatedEmitter scalar_region;
    // todo: this clone_with_new_inputs is irrelevant
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return std::make_shared<TileScheduler>(vector_region, scalar_region);
    }
    const void *compile_params;
};

} // namespace op
} // namespace snippets
} // namespace ngraph
