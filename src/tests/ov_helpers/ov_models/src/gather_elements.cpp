// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather_elements.hpp"

#include <memory>
#include <numeric>
#include <vector>

#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeGatherElements(const Output<Node>& dataNode,
                                         const Shape& indicesShape,
                                         const element::Type& indicesType,
                                         const int axis) {
    const auto& dataShape = dataNode.get_shape();
    int posAxis = axis;
    if (posAxis < 0)
        posAxis += dataShape.size();
    const auto axisDim = dataShape[posAxis];
    const auto indicesSize =
        std::accumulate(begin(indicesShape), end(indicesShape), 1ull, std::multiplies<std::size_t>{});

    auto indicesValues = NGraphFunctions::Utils::generateVector<element::Type_t::i32>(indicesSize, axisDim - 1, 0);
    auto indicesNode = ov::op::v0::Constant::create(indicesType, indicesShape, indicesValues);

    auto gatherElNode = std::make_shared<ov::op::v6::GatherElements>(dataNode, indicesNode, axis);
    gatherElNode->set_friendly_name("GatherElements");

    return gatherElNode;
}

}  // namespace builder
}  // namespace ngraph
