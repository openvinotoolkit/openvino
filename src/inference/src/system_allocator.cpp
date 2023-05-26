// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "system_allocator.hpp"

namespace InferenceEngine {
IE_SUPPRESS_DEPRECATED_START

INFERENCE_ENGINE_API_CPP(std::shared_ptr<IAllocator>) CreateDefaultAllocator() noexcept {
    try {
        return std::make_shared<SystemMemoryAllocator>();
    } catch (...) {
        return nullptr;
    }
}

}  // namespace InferenceEngine
