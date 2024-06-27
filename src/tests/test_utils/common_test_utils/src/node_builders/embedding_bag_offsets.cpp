// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/embedding_bag_offsets.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/embeddingbag_offsets.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_embedding_bag_offsets(const element::Type& data_type,
                                                     const ov::element::Type& indices_type,
                                                     const ov::Output<Node>& emb_table_node,
                                                     const std::vector<size_t>& indices,
                                                     const std::vector<size_t>& offsets,
                                                     size_t default_index,
                                                     bool with_weights,
                                                     bool with_default_index,
                                                     ov::op::util::EmbeddingBagOffsetsBase::Reduction reduction) {
    ov::Shape i_shape = {indices.size()};
    auto indices_node = std::make_shared<ov::op::v0::Constant>(indices_type, i_shape, indices);
    ov::Shape o_shape = {offsets.size()};
    auto offsetsNode = std::make_shared<ov::op::v0::Constant>(indices_type, o_shape, offsets);

    std::shared_ptr<Node> embBag;
    if (with_default_index) {
        auto defIdxNode = std::make_shared<ov::op::v0::Constant>(indices_type, ov::Shape{}, default_index);
        if (with_weights) {
            auto tensor = create_and_fill_tensor(data_type, ov::Shape{indices.size()});
            auto weights_node = std::make_shared<ov::op::v0::Constant>(tensor);

            embBag = std::make_shared<ov::op::v15::EmbeddingBagOffsets>(emb_table_node,
                                                                        indices_node,
                                                                        offsetsNode,
                                                                        defIdxNode,
                                                                        weights_node,
                                                                        reduction);
        } else {
            embBag = std::make_shared<ov::op::v15::EmbeddingBagOffsets>(emb_table_node,
                                                                        indices_node,
                                                                        offsetsNode,
                                                                        defIdxNode,
                                                                        reduction);
        }
    } else {
        if (with_weights) {
            auto defIdxNode = ov::op::v0::Constant::create(indices_type, ov::Shape{}, {-1});
            auto tensor = create_and_fill_tensor(data_type, ov::Shape{indices.size()});
            auto weights_node = std::make_shared<ov::op::v0::Constant>(tensor);
            embBag = std::make_shared<ov::op::v15::EmbeddingBagOffsets>(emb_table_node,
                                                                        indices_node,
                                                                        offsetsNode,
                                                                        defIdxNode,
                                                                        weights_node,
                                                                        reduction);
        } else {
            embBag = std::make_shared<ov::op::v15::EmbeddingBagOffsets>(emb_table_node,
                                                                        indices_node,
                                                                        offsetsNode,
                                                                        reduction);
        }
    }
    return embBag;
}
}  // namespace utils
}  // namespace test
}  // namespace ov
