// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <atomic>
#include <thread>

#include "cpu_memory.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "common_test_utils/test_assertions.hpp"

using namespace ov::intel_cpu;

TEST(MemoryTest, SedDataCheck) {
    GTEST_SKIP();
}

TEST(MemoryTest, SedDataWithAutoPadCheck) {
    GTEST_SKIP();
}

TEST(StaticMemoryTest, UnsupportedDnnlPrecision) {
    // in the context of this test, unsupported precision means a precision unsupported by oneDNN
    const dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    CpuBlockedMemoryDesc memDescSupportedPrc(ov::element::f32, {5, 4, 7, 10});
    MemoryPtr testMemory;
    OV_ASSERT_NO_THROW(testMemory = std::make_shared<StaticMemory>(memDescSupportedPrc));
    ASSERT_TRUE(testMemory->isDefined());
    dnnl::memory dnnl_memory;
    void* raw_data_ptr = nullptr;
    OV_ASSERT_NO_THROW(raw_data_ptr = testMemory->getData());
    ASSERT_FALSE(nullptr == raw_data_ptr);
    OV_ASSERT_NO_THROW(dnnl_memory = DnnlExtensionUtils::createMemoryPrimitive(testMemory, eng));
    ASSERT_TRUE(dnnl_memory);

    CpuBlockedMemoryDesc memDescUnSupportedPrc(ov::element::i64, {5, 4, 7, 10});
    OV_ASSERT_NO_THROW(testMemory = std::make_shared<StaticMemory>(memDescUnSupportedPrc));
    ASSERT_TRUE(testMemory->isDefined());
    raw_data_ptr = nullptr;
    OV_ASSERT_NO_THROW(raw_data_ptr = testMemory->getData());
    ASSERT_FALSE(nullptr == raw_data_ptr);
    dnnl_memory = dnnl::memory();
    ASSERT_THROW(dnnl_memory = DnnlExtensionUtils::createMemoryPrimitive(testMemory, eng), ov::Exception);
    ASSERT_FALSE(dnnl_memory);
}
