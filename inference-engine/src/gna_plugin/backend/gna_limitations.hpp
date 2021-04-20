// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace GNAPluginNS {
namespace GNALimitations {

constexpr uint32_t convMinFiltersNum = 4;
constexpr uint32_t convMaxFiltersNum = 65532;
constexpr uint32_t convFiltersNumDivider = 4;
constexpr uint32_t convEachKernelByteAlignment = 16;
constexpr uint32_t noOfInputsDivisor = 8;
constexpr uint32_t noOfInputsLowPrecDivisor = 16;

}
} // namespace GNAPluginNS
