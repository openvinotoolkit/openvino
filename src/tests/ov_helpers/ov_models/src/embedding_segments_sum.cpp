// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/embedding_segments_sum.hpp"

#include <memory>
#include <vector>

#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeEmbeddingSegmentsSum(const element::Type& dataType,
                                               const element::Type& indicesType,
                                               const ov::Output<Node>& embTableNode,
                                               const std::vector<size_t>& indices,
                                               const std::vector<size_t>& segment_ids,
                                               size_t num_segments,
                                               size_t default_index,
                                               bool with_weights,
                                               bool with_default_index) {
    std::vector<size_t> i_shape = {indices.size()};
    auto indicesNode = std::make_shared<ov::op::v0::Constant>(indicesType, i_shape, indices);
    std::vector<size_t> o_shape = {segment_ids.size()};
    auto segmentIdNode = std::make_shared<ov::op::v0::Constant>(indicesType, o_shape, segment_ids);
    std::vector<size_t> shape_0 = {};
    auto segmentNumNode = std::make_shared<ov::op::v0::Constant>(indicesType, shape_0, num_segments);

    std::shared_ptr<Node> embBag;
    if (with_default_index) {
        auto defIdxNode = std::make_shared<ov::op::v0::Constant>(indicesType, shape_0, default_index);
        if (with_weights) {
            auto weightsNode = makeConstant<float>(dataType, {indices.size()}, {}, true);

            embBag = std::make_shared<ov::op::v3::EmbeddingSegmentsSum>(embTableNode,
                                                                        indicesNode,
                                                                        segmentIdNode,
                                                                        segmentNumNode,
                                                                        defIdxNode,
                                                                        weightsNode);
        } else {
            embBag = std::make_shared<ov::op::v3::EmbeddingSegmentsSum>(embTableNode,
                                                                        indicesNode,
                                                                        segmentIdNode,
                                                                        segmentNumNode,
                                                                        defIdxNode);
        }
    } else {
        embBag = std::make_shared<ov::op::v3::EmbeddingSegmentsSum>(embTableNode,
                                                                    indicesNode,
                                                                    segmentIdNode,
                                                                    segmentNumNode);
    }
    return embBag;
}

}  // namespace builder
}  // namespace ngraph
