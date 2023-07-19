// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
// Created by user on 20.10.16.
//

#pragma once

#include <gmock/gmock.h>

#include "ie_allocator.hpp"

IE_SUPPRESS_DEPRECATED_START
class MockAllocator : public InferenceEngine::IAllocator {
public:
    MOCK_METHOD(void*, lock, (void*, InferenceEngine::LockOp), (noexcept));
    MOCK_METHOD(void, unlock, (void*), (noexcept));
    MOCK_METHOD(void*, alloc, (size_t), (noexcept));
    MOCK_METHOD(bool, free, (void*), (noexcept));  // NOLINT(readability/casting)
};
IE_SUPPRESS_DEPRECATED_END
