// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_fake_convert(const ov::Output<ov::Node>& in,
                                            const ov::Output<ov::Node>& scale,
                                            const ov::Output<ov::Node>& shift,
                                            ov::element::Type destination_type);

std::shared_ptr<ov::Node> make_fake_convert(const ov::Output<Node>& in,
                                            const ov::Output<ov::Node>& scale,
                                            ov::element::Type destination_type);
}  // namespace utils
}  // namespace test
}  // namespace ov
