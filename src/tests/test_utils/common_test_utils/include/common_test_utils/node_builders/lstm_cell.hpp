// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/node.hpp"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_lstm(
    const ov::OutputVector& in,
    const std::vector<ov::Shape>& constants,
    std::size_t hidden_size,
    const std::vector<std::string>& activations = std::vector<std::string>{"sigmoid", "tanh", "tanh"},
    const std::vector<float>& activations_alpha = {},
    const std::vector<float>& activations_beta = {},
    float clip = 0.f,
    bool make_sequence = false,
    ov::op::RecurrentSequenceDirection direction = ov::op::RecurrentSequenceDirection::FORWARD,
    ov::test::utils::SequenceTestsMode mode = ov::test::utils::SequenceTestsMode::PURE_SEQ,
    float WRB_range = 0.f);

}  // namespace utils
}  // namespace test
}  // namespace ov