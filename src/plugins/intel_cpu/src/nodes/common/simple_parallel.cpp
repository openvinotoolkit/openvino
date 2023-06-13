// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "simple_parallel.h"

#include <vector>
#include <string>
#include "ie_parallel.hpp"

namespace ov {
namespace cpu {

size_t getTotalThreads() {
    return parallel_get_max_threads();
}

void TrySimpleParallelFor(const std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn) {
    parallel_for(total, fn);
}

};  // namespace cpu
};  // namespace ov