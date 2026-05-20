// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>


namespace memory_tests::gpu {

struct Sample {
    // memory size in kb
    int64_t local_used = -1;
    int64_t local_total = -1;
    int64_t nonlocal_used = -1;
    int64_t nonlocal_total = -1;
};


enum class InitStatus {
    SUCCESS,
    SUBSYSTEM_UNAVAILABLE,
    SUBSYSTEM_UNSUPPORTED,
    GPU_NOT_FOUND,
};


InitStatus init();

Sample sample();

}  // namespace memory_tests::gpu
