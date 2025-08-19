// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multi_cache.h"

#include <atomic>

namespace ov::intel_cpu {

std::atomic_size_t MultiCache::_typeIdCounter{0};

}  // namespace ov::intel_cpu
