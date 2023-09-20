// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "openvino/op/embeddingbag_packedsum.hpp"
#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeEmbeddingBagPackedSum(const element::Type& dataType,
                                                const element::Type& indicesType,
                                                const ov::Output<Node>& embTableNode,
                                                const std::vector<std::vector<size_t>>& indices,
                                                bool with_weights) {
    std::vector<size_t> i_shape({indices.size(), indices[0].size()});
    size_t i_size = ov::shape_size(i_shape);
    std::vector<size_t> i_values(i_size);
    for (int i = 0; i < indices.size(); i++)
        memcpy(i_values.data() + indices[0].size() * i, indices[i].data(), indices[0].size() * sizeof(size_t));
    auto indicesNode = std::make_shared<ov::op::v0::Constant>(indicesType, i_shape, i_values);

    std::shared_ptr<Node> embBag;
    if (with_weights) {
        auto weightsNode = makeConstant<float>(dataType, i_shape, {}, true);

        embBag = std::make_shared<ov::op::v3::EmbeddingBagPackedSum>(embTableNode, indicesNode, weightsNode);
    } else {
        embBag = std::make_shared<ov::op::v3::EmbeddingBagPackedSum>(embTableNode, indicesNode);
    }
    return embBag;
}

}  // namespace builder
}  // namespace ngraph
