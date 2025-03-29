// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/embedding_bag_packed.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/embeddingbag_packed.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_embedding_bag_packed(const ov::element::Type& data_type,
                                                    const ov::element::Type& indices_type,
                                                    const ov::Output<Node>& emb_table_node,
                                                    const std::vector<std::vector<size_t>>& indices,
                                                    bool with_weights,
                                                    ov::op::util::EmbeddingBagPackedBase::Reduction reduction) {
    ov::Shape i_shape({indices.size(), indices[0].size()});
    size_t i_size = ov::shape_size(i_shape);
    std::vector<size_t> i_values(i_size);
    for (int i = 0; i < indices.size(); i++)
        memcpy(i_values.data() + indices[0].size() * i, indices[i].data(), indices[0].size() * sizeof(size_t));
    auto indicesNode = std::make_shared<ov::op::v0::Constant>(indices_type, i_shape, i_values);

    std::shared_ptr<Node> embBag;
    if (with_weights) {
        auto tensor = create_and_fill_tensor(data_type, i_shape);
        auto weights_node = std::make_shared<ov::op::v0::Constant>(tensor);

        embBag =
            std::make_shared<ov::op::v15::EmbeddingBagPacked>(emb_table_node, indicesNode, weights_node, reduction);
    } else {
        embBag = std::make_shared<ov::op::v15::EmbeddingBagPacked>(emb_table_node, indicesNode, reduction);
    }
    return embBag;
}
}  // namespace utils
}  // namespace test
}  // namespace ov
