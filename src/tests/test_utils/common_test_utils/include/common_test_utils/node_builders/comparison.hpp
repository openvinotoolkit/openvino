// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_enums.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_comparison(const ov::Output<Node>& in0,
                                          const ov::Output<Node>& in1,
                                          ov::test::utils::ComparisonTypes comparison_type);
}  // namespace utils
}  // namespace test
}  // namespace ov