// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_enums.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_fully_connected(const ov::Output<Node>& in,
                                               const ov::element::Type& type,
                                               const size_t output_size,
                                               bool addBias = true,
                                               const ov::Shape& weights_shape = {},
                                               const std::vector<float>& weights = {},
                                               const std::vector<float>& bias_weights = {});
}  // namespace utils
}  // namespace test
}  // namespace ov