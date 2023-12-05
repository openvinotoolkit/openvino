// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather_nd.hpp"

#include <memory>
#include <numeric>
#include <vector>

#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeGatherND(const ov::Output<Node>& dataNode,
                                   const ov::Shape& indicesShape,
                                   const element::Type& indicesType,
                                   const std::size_t batchDims) {
    const auto indices = [&] {
        const auto& dataShape = dataNode.get_shape();
        const auto indicesCount =
            std::accumulate(begin(indicesShape), prev(end(indicesShape)), 1ull, std::multiplies<std::size_t>{});
        const auto sliceRank = indicesShape.back();

        const auto maxDim = *std::max_element(begin(dataShape), end(dataShape));

        auto indicesValues =
            NGraphFunctions::Utils::generateVector<element::Type_t::i32>(indicesCount * sliceRank, maxDim, 0);
        auto indicesData = indicesValues.data();
        for (int i = 0; i < indicesCount; i++) {
            for (int dim = 0; dim < sliceRank; dim++) {
                indicesData[0] = indicesData[0] % dataShape[dim + batchDims];
                indicesData++;
            }
        }
        return op::v0::Constant::create(indicesType, indicesShape, indicesValues);
    }();

    auto gatherNdNode = std::make_shared<ov::op::v5::GatherND>(dataNode, indices, batchDims);
    gatherNdNode->set_friendly_name("GatherND");

    return gatherNdNode;
}

std::shared_ptr<Node> makeGatherND8(const ov::Output<Node>& dataNode,
                                    const ov::Shape& indicesShape,
                                    const element::Type& indicesType,
                                    const std::size_t batchDims) {
    const auto indices = [&] {
        const auto& dataShape = dataNode.get_shape();
        const auto indicesCount =
            std::accumulate(begin(indicesShape), prev(end(indicesShape)), 1ull, std::multiplies<std::size_t>{});
        const auto sliceRank = indicesShape.back();

        const auto maxDim = *std::max_element(begin(dataShape), end(dataShape));

        auto indicesValues =
            NGraphFunctions::Utils::generateVector<element::Type_t::i32>(indicesCount * sliceRank, maxDim, 0);
        auto indicesData = indicesValues.data();
        for (int i = 0; i < indicesCount; i++) {
            for (int dim = 0; dim < sliceRank; dim++) {
                indicesData[0] = indicesData[0] % dataShape[dim + batchDims];
                indicesData++;
            }
        }
        return op::v0::Constant::create(indicesType, indicesShape, indicesValues);
    }();

    auto gatherNdNode = std::make_shared<ov::op::v8::GatherND>(dataNode, indices, batchDims);
    gatherNdNode->set_friendly_name("GatherND");

    return gatherNdNode;
}
}  // namespace builder
}  // namespace ngraph
