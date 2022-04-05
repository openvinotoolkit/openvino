// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeEmbeddingBagOffsetsSum(
                                      const element::Type& dataType,
                                      const element::Type& indicesType,
                                      const ngraph::Output<Node>& embTableNode,
                                      const std::vector<size_t>& indices,
                                      const std::vector<size_t>& offsets,
                                      size_t default_index,
                                      bool with_weights,
                                      bool with_default_index) {
    std::vector<size_t> i_shape = {indices.size()};
    auto indicesNode = std::make_shared<ngraph::opset1::Constant>(indicesType, i_shape, indices);
    std::vector<size_t> o_shape = {offsets.size()};
    auto offsetsNode = std::make_shared<ngraph::opset1::Constant>(indicesType, o_shape, offsets);

    std::shared_ptr<Node> embBag;
    if (with_default_index) {
        std::vector<size_t> d_shape = {};
        auto defIdxNode = std::make_shared<ngraph::opset1::Constant>(indicesType, d_shape, default_index);
        if (with_weights) {
            auto weightsNode = makeConstant<float>(dataType, {indices.size()}, {}, true);

            embBag = std::make_shared<opset3::EmbeddingBagOffsetsSum>(
                embTableNode, indicesNode, offsetsNode, defIdxNode, weightsNode);
        } else {
            embBag = std::make_shared<opset3::EmbeddingBagOffsetsSum>(
                embTableNode, indicesNode, offsetsNode, defIdxNode);
        }
    } else {
        embBag = std::make_shared<opset3::EmbeddingBagOffsetsSum>(
            embTableNode, indicesNode, offsetsNode);
    }
    return embBag;
}

}  // namespace builder
}  // namespace ngraph
