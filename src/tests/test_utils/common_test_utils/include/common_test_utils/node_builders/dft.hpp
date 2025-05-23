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
std::shared_ptr<ov::Node> make_dft(ov::ParameterVector& parameters,
                                   const std::vector<int64_t>& axes,
                                   const std::vector<int64_t>& signal_size,
                                   const ov::test::utils::DFTOpType op_type,
                                   utils::InputLayerType axes_in_type = utils::InputLayerType::CONSTANT,
                                   utils::InputLayerType size_in_type = utils::InputLayerType::PARAMETER);
}  // namespace utils
}  // namespace test
}  // namespace ov