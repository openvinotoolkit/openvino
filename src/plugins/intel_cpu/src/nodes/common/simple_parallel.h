// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <string>
#include <functional>

#ifdef OV_CPU_WITH_LLMDNN

namespace llmdnn {

size_t get_total_threads();
void simple_parallel_for(const size_t total, const std::function<void(size_t)>& fn);

}  // namespace llmdnn

#endif