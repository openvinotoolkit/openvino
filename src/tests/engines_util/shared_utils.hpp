// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <vector>

namespace ngraph {
namespace test {
testing::AssertionResult compare_with_tolerance(const std::vector<float>& expected_results,
                                                const std::vector<float>& results,
                                                const float tolerance);
}
}  // namespace ngraph
