// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
// Class for pretty-logging.
//

#pragma once

#include <iostream>

namespace intel_npu {

namespace utils {

constexpr std::size_t STANDARD_PAGE_SIZE = 4096;

static inline bool memory_and_size_aligned_to_standard_page_size(void* addr, size_t size) {
    auto addr_int = reinterpret_cast<uintptr_t>(addr);

    // addr is aligned to standard page size
    return (addr_int % STANDARD_PAGE_SIZE == 0) && (size % STANDARD_PAGE_SIZE == 0);
}
}

}  // namespace intel_npu
