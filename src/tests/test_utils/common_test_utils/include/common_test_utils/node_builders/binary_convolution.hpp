// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_binary_convolution(const ov::Output<Node>& in,
                                                  const std::vector<size_t>& filterSize,
                                                  const std::vector<size_t>& strides,
                                                  const std::vector<ptrdiff_t>& padsBegin,
                                                  const std::vector<ptrdiff_t>& padsEnd,
                                                  const std::vector<size_t>& dilations,
                                                  const ov::op::PadType& autoPad,
                                                  size_t numOutChannels,
                                                  float padValue,
                                                  const std::vector<int8_t>& filterWeihgts = {});
}  // namespace utils
}  // namespace test
}  // namespace ov
