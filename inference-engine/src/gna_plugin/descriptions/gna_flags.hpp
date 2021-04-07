// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace GNAPluginNS {
struct GNAFlags {
    uint8_t gna_lib_async_threads_num = 1;

    bool compact_mode = false;
    bool exclusive_async_requests = false;
    bool uniformPwlDesign = false;
    float pwlMaxErrorPercent = 1.0f;
    bool gna_openmp_multithreading = false;
    bool sw_fp32 = false;
    bool fake_quantized = false;
    bool performance_counting = false;
    bool input_low_precision = false;
};
}  // namespace GNAPluginNS
