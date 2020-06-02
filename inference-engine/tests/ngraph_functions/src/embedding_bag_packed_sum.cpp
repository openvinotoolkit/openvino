// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeEmbeddingBagPackedSum(
                                      const element::Type& dataType,
                                      const element::Type& indicesType,
                                      const ngraph::Output<Node>& embTableNode,
                                      const std::vector<std::vector<size_t>>& indices,
                                      bool  with_weights) {
    std::vector<size_t> i_shape({indices.size(), indices[0].size()});
    size_t i_size = ngraph::shape_size(i_shape);
    std::vector<size_t> i_values(i_size);
    for (int i = 0; i < indices.size(); i++)
        memcpy(i_values.data() + indices[0].size() * i, indices[i].data(), indices[0].size() * sizeof(size_t));
    auto indicesNode = std::make_shared<ngraph::opset1::Constant>(indicesType, i_shape, i_values);

    std::shared_ptr<Node> embBag;
    if (with_weights) {
        auto weightsNode = makeConstant(dataType, i_shape, {}, true);

        embBag = std::make_shared<opset3::EmbeddingBagPackedSum>(
            embTableNode, indicesNode, weightsNode);
    } else {
        embBag = std::make_shared<opset3::EmbeddingBagPackedSum>(
            embTableNode, indicesNode);
    }
    return embBag;
}

}  // namespace builder
}  // namespace ngraph
