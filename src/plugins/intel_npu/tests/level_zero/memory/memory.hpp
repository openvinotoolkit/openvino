// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "zero_memory.hpp"

using namespace intel_npu;

class MemoryAllocator : public ::testing::TestWithParam<size_t> {
protected:
    void SetUp() override;

    void TearDown() override;

public:
    std::shared_ptr<zeroMemory::HostMemSharedAllocator> sharedAllocator;

    std::shared_ptr<zeroMemory::HostMemAllocator> allocator;

    static std::string getTestCaseName(testing::TestParamInfo<size_t> obj) {
        return std::to_string(obj.param);
    }
};
