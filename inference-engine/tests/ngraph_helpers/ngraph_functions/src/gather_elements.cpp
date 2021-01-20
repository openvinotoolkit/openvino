// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <numeric>
#include <vector>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeGatherElements(const ngraph::Output<Node>& dataNode,
                                         const element::Type& indicesType,
                                         const std::size_t axis,
                                         const std::size_t indices_axis_dim) {
    const auto indices = [&] {
        const auto& dataShape = dataNode.get_shape();
        const auto indicesCount = std::accumulate(begin(dataShape), end(dataShape),
                                                  1ull, std::multiplies<std::size_t>{}) / dataShape[axis] * indices_axis_dim;

        auto indicesShape = dataShape;
        indicesShape[axis] = indices_axis_dim;
        const auto dim0 = dataShape[axis];

        auto indicesValues = NGraphFunctions::Utils::generateVector<element::Type_t::i32>(indicesCount, dim0 - 1, 0);

        return opset5::Constant::create(indicesType, indicesShape, indicesValues);
    }();

    auto gatherElementsNode = std::make_shared<opset6::GatherElements>(dataNode, indices, axis);
    gatherElementsNode->set_friendly_name("GatherElements");

    return gatherElementsNode;
}

}  // namespace builder
}  // namespace ngraph
