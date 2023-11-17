// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/node.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_embedding_bag_packed_sum(const ov::element::Type& data_type,
                                                        const ov::element::Type& indices_type,
                                                        const ov::Output<Node>& emb_table_node,
                                                        const std::vector<std::vector<size_t>>& indices,
                                                        bool with_weights);
}  // namespace utils
}  // namespace test
}  // namespace ov
