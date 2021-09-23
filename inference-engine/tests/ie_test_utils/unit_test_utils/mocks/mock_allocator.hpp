// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
// Created by user on 20.10.16.
//

#pragma once

#include <gmock/gmock.h>

#include "ie_allocator.hpp"
#include "openvino/runtime/allocator.hpp"

class MockAllocator : public InferenceEngine::IAllocator {
public:
    MOCK_METHOD(void*, lock, (void*, InferenceEngine::LockOp), (noexcept));
    MOCK_METHOD(void, unlock, (void *), (noexcept));
    MOCK_METHOD(void*, alloc, (size_t), (noexcept));
    MOCK_METHOD(bool, free, (void*), (noexcept));  // NOLINT(readability/casting)
};


class OVMockAllocator : public ov::AllocatorImpl {
public:
    MOCK_METHOD(void*, allocate, (size_t, size_t), ());
    MOCK_METHOD(void, deallocate, (void*, size_t, size_t), ());  // NOLINT(readability/casting)
    MOCK_METHOD(bool, is_equal, (const ov::AllocatorImpl&), (const, noexcept));  // NOLINT(readability/casting)
};
