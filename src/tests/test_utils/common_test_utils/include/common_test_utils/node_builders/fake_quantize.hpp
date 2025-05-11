// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_fake_quantize(const ov::Output<ov::Node>& in,
                                             const ov::element::Type& type,
                                             std::size_t levels,
                                             std::vector<size_t> constShapes,
                                             const std::vector<float>& inputLowData,
                                             const std::vector<float>& inputHighData,
                                             const std::vector<float>& outputLowData,
                                             const std::vector<float>& outputHighData);

std::shared_ptr<ov::Node> make_fake_quantize(const ov::Output<Node>& in,
                                             const ov::element::Type& type,
                                             std::size_t levels,
                                             std::vector<size_t> constShapes);
}  // namespace utils
}  // namespace test
}  // namespace ov
