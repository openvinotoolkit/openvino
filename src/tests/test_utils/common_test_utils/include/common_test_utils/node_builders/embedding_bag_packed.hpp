// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/util/embeddingbag_packed_base.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_embedding_bag_packed(const ov::element::Type& data_type,
                                                    const ov::element::Type& indices_type,
                                                    const ov::Output<Node>& emb_table_node,
                                                    const std::vector<std::vector<size_t>>& indices,
                                                    bool with_weights,
                                                    ov::op::util::EmbeddingBagPackedBase::Reduction reduction);
}  // namespace utils
}  // namespace test
}  // namespace ov
