// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#pragma once

#include "common_test_utils/test_enums.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_activation(const ov::Output<Node>& in,
                                          const element::Type& type,
                                          ov::test::utils::ActivationTypes activation_type,
                                          ov::Shape in_shape = {},
                                          std::vector<float> constants_value = {});

std::shared_ptr<ov::Node> make_activation(const ov::ParameterVector& parameters,
                                          const element::Type& type,
                                          ov::test::utils::ActivationTypes activation_type);
}  // namespace utils
}  // namespace test
}  // namespace ov
