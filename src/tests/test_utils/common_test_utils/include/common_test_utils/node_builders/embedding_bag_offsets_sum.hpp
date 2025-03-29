// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_embedding_bag_offsets_sum(const element::Type& data_type,
                                                         const ov::element::Type& indices_type,
                                                         const ov::Output<Node>& emb_table_node,
                                                         const std::vector<size_t>& indices,
                                                         const std::vector<size_t>& offsets,
                                                         size_t default_index,
                                                         bool with_weights,
                                                         bool with_default_index);
}  // namespace utils
}  // namespace test
}  // namespace ov
