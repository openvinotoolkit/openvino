// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_enums.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_extremum(const ov::Output<Node>& in0,
                                        const ov::Output<Node>& in1,
                                        ov::test::utils::MinMaxOpType extremum_type);
}  // namespace utils
}  // namespace test
}  // namespace ov
