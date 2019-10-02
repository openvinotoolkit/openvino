// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "system_allocator.hpp"

namespace InferenceEngine {

IAllocator* CreateDefaultAllocator() noexcept {
    try {
        return new SystemMemoryAllocator();
    } catch (...) {
        return nullptr;
    }
}

}  // namespace InferenceEngine
