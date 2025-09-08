//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <sstream>
#include <vector>

namespace utils {

float runPSNRMetric(std::vector<std::vector<float>>& actOutput, std::vector<std::vector<float>>& refOutput,
                    const size_t imgHeight, const size_t imgWidth, int scaleBorder, bool normalizedImage);

}  // namespace utils
