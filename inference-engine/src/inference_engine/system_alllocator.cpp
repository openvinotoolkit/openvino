// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "system_alllocator.hpp"

INFERENCE_ENGINE_API(InferenceEngine::IAllocator*)CreateDefaultAllocator() noexcept {
    try {
        return new SystemMemoryAllocator();
    }catch (...) {
        return nullptr;
    }
}