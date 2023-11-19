// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/node.hpp"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_augru(
    const OutputVector& in,
    const std::vector<ov::Shape>& constants,
    std::size_t hidden_size,
    bool make_sequence = false,
    ov::op::RecurrentSequenceDirection direction = ov::op::RecurrentSequenceDirection::FORWARD,
    ov::test::utils::SequenceTestsMode mode = ov::test::utils::SequenceTestsMode::PURE_SEQ);
}  // namespace utils
}  // namespace test
}  // namespace ov