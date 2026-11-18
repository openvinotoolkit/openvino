// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstdint>

namespace ov::intel_gpu::detail {

inline int64_t estimate_max_batch_size(int64_t available_device_mem,
                                       int64_t constant_mem_usage,
                                       int64_t general_mem_usage,
                                       uint32_t base_batch_size,
                                       uint32_t n_streams) {
    // Without a positive general-memory estimate, a one-byte per-batch divisor can produce a
    // meaningless large metric. Keep the conservative default used by the caller's other fallbacks.
    if (general_mem_usage <= 0 || base_batch_size == 0 || n_streams == 0) {
        return 1;
    }

    const int64_t mem_for_general = std::max<int64_t>(1, available_device_mem - constant_mem_usage);
    const int64_t mem_per_batch = std::max<int64_t>(1, general_mem_usage / base_batch_size);
    return mem_for_general / (mem_per_batch * n_streams);
}

}  // namespace ov::intel_gpu::detail
