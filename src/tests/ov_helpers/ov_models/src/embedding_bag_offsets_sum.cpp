// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "openvino/op/embeddingbag_offsets_sum.hpp"
#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeEmbeddingBagOffsetsSum(const element::Type& dataType,
                                                 const element::Type& indicesType,
                                                 const ov::Output<Node>& embTableNode,
                                                 const std::vector<size_t>& indices,
                                                 const std::vector<size_t>& offsets,
                                                 size_t default_index,
                                                 bool with_weights,
                                                 bool with_default_index) {
    std::vector<size_t> i_shape = {indices.size()};
    auto indicesNode = std::make_shared<ov::op::v0::Constant>(indicesType, i_shape, indices);
    std::vector<size_t> o_shape = {offsets.size()};
    auto offsetsNode = std::make_shared<ov::op::v0::Constant>(indicesType, o_shape, offsets);

    std::shared_ptr<Node> embBag;
    if (with_default_index) {
        std::vector<size_t> d_shape = {};
        auto defIdxNode = std::make_shared<ov::op::v0::Constant>(indicesType, d_shape, default_index);
        if (with_weights) {
            auto weightsNode = makeConstant<float>(dataType, {indices.size()}, {}, true);

            embBag = std::make_shared<ov::op::v3::EmbeddingBagOffsetsSum>(embTableNode,
                                                                          indicesNode,
                                                                          offsetsNode,
                                                                          defIdxNode,
                                                                          weightsNode);
        } else {
            embBag = std::make_shared<ov::op::v3::EmbeddingBagOffsetsSum>(embTableNode,
                                                                          indicesNode,
                                                                          offsetsNode,
                                                                          defIdxNode);
        }
    } else {
        embBag = std::make_shared<ov::op::v3::EmbeddingBagOffsetsSum>(embTableNode, indicesNode, offsetsNode);
    }
    return embBag;
}

}  // namespace builder
}  // namespace ngraph
