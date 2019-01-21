// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
// Created by user on 20.10.16.
//

#pragma once

#include <gmock/gmock.h>

#include "ie_allocator.hpp"


class MockAllocator : public InferenceEngine::IAllocator
{
public:
    MOCK_QUALIFIED_METHOD0(Release, noexcept, void());
    MOCK_QUALIFIED_METHOD2(lock, noexcept, void*(void*, InferenceEngine::LockOp));
    MOCK_QUALIFIED_METHOD1(unlock, noexcept, void(void * ));
    MOCK_QUALIFIED_METHOD1(alloc, noexcept, void*(size_t));
    MOCK_QUALIFIED_METHOD1(free, noexcept, bool(void*));
};

