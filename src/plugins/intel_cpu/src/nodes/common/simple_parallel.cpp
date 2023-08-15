// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "simple_parallel.h"

#include <vector>
#include <string>
#include "ie_parallel.hpp"

#ifdef OV_CPU_WITH_LLMDNN

namespace llmdnn {

size_t get_total_threads() {
    return parallel_get_max_threads();
}

void simple_parallel_for(const size_t total, const std::function<void(size_t)>& fn) {
    ov::parallel_for(total, fn);
}

}  // namespace llmdnn

#endif