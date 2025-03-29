// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_gather_nd(const ov::Output<Node>& data_node,
                                         const ov::Shape& indices_shape,
                                         const ov::element::Type& indices_type,
                                         const std::size_t batch_dims);

std::shared_ptr<ov::Node> make_gather_nd8(const ov::Output<Node>& data_node,
                                          const ov::Shape& indices_shape,
                                          const ov::element::Type& indices_type,
                                          const std::size_t batch_dims);
}  // namespace utils
}  // namespace test
}  // namespace ov