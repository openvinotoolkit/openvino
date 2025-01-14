// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_group_convolution(const ov::Output<Node>& in,
                                                 const ov::element::Type& type,
                                                 const std::vector<size_t>& filterSize,
                                                 const std::vector<size_t>& strides,
                                                 const std::vector<ptrdiff_t>& padsBegin,
                                                 const std::vector<ptrdiff_t>& padsEnd,
                                                 const std::vector<size_t>& dilations,
                                                 const ov::op::PadType& autoPad,
                                                 size_t numOutChannels,
                                                 size_t numGroups,
                                                 bool addBiases = false,
                                                 const std::vector<float>& filterWeights = {},
                                                 const std::vector<float>& biasesWeights = {});

std::shared_ptr<ov::Node> make_group_convolution(const ov::Output<Node>& in,
                                                 const ov::Output<Node>& weights,
                                                 const ov::element::Type& type,
                                                 const std::vector<size_t>& strides,
                                                 const std::vector<ptrdiff_t>& padsBegin,
                                                 const std::vector<ptrdiff_t>& padsEnd,
                                                 const std::vector<size_t>& dilations,
                                                 const ov::op::PadType& autoPad,
                                                 bool addBiases = false,
                                                 const std::vector<float>& biasesWeights = {});
}  // namespace utils
}  // namespace test
}  // namespace ov
