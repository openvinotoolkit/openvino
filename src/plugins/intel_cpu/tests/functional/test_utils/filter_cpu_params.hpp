// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_test_utils.hpp"
#include <vector>

namespace CPUTestUtils {
std::vector<CPUSpecificParams> filterCPUSpecificParams(const std::vector<CPUSpecificParams>& paramsVector);
} // namespace CPUTestUtils