// Copyright (C) 2018-2024 Intel Corporation
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

TEST(MemoryTest, ConcurrentGetPrimitive) {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::memory dnnl_mem1;
    dnnl::memory dnnl_mem2;
    auto desc = std::make_shared<CpuBlockedMemoryDesc>(ov::element::f32, Shape{10, 2});
    Memory cpu_mem1(eng, desc);

    std::atomic<bool> lock{true};

    std::thread worker1([&](){
        while (lock.load()) {}
        dnnl_mem1 = cpu_mem1.getPrimitive();
    });

    std::thread worker2([&](){
        while (lock.load()) {}
        dnnl_mem2 = cpu_mem1.getPrimitive();
    });

    lock.store(false);

    worker1.join();
    worker2.join();
    ASSERT_EQ(dnnl_mem1.get_data_handle(), cpu_mem1.getData());
    ASSERT_EQ(dnnl_mem1, dnnl_mem2);
}

TEST(MemoryTest, ConcurrentResizeGetPrimitive) {
    constexpr size_t number_of_attempts = 10; //just to increase the probability of a collision
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    for (size_t i = 0; i < number_of_attempts; ++i) {
        dnnl::memory dnnl_mem;
        auto desc = std::make_shared<CpuBlockedMemoryDesc>(ov::element::f32, Shape{10, 2});
        Memory cpu_mem1(eng, desc);
        Memory cpu_mem2(eng, desc, cpu_mem1.getMemoryBlock());
        auto desc2 = std::make_shared<CpuBlockedMemoryDesc>(ov::element::f32, Shape{10, 20});

        std::atomic<bool> lock{true};

        std::thread worker1([&](){
            while (lock.load()) {}
            dnnl_mem = cpu_mem1.getPrimitive();
        });

        std::thread worker2([&](){
            while (lock.load()) {}
            cpu_mem2.redefineDesc(desc2);
        });

        lock.store(false);

        worker1.join();
        worker2.join();
        ASSERT_EQ(dnnl_mem.get_data_handle(), cpu_mem2.getData());
    }
}

TEST(StaticMemoryTest, UnsupportedDnnlPrecision) {
    // in the context of this test, unsupported precision means a precision unsupported by oneDNN
    const dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    CpuBlockedMemoryDesc memDescSupportedPrc(ov::element::f32, {5, 4, 7, 10});
    MemoryPtr testMemory;
    OV_ASSERT_NO_THROW(testMemory = std::make_shared<StaticMemory>(eng, memDescSupportedPrc));
    ASSERT_TRUE(testMemory->isDefined());
    dnnl::memory dnnl_memory;
    void* raw_data_ptr = nullptr;
    OV_ASSERT_NO_THROW(raw_data_ptr = testMemory->getData());
    ASSERT_FALSE(nullptr == raw_data_ptr);
    OV_ASSERT_NO_THROW(dnnl_memory = testMemory->getPrimitive());
    ASSERT_TRUE(dnnl_memory);

    CpuBlockedMemoryDesc memDescUnSupportedPrc(ov::element::i64, {5, 4, 7, 10});
    OV_ASSERT_NO_THROW(testMemory = std::make_shared<StaticMemory>(eng, memDescUnSupportedPrc));
    ASSERT_TRUE(testMemory->isDefined());
    raw_data_ptr = nullptr;
    OV_ASSERT_NO_THROW(raw_data_ptr = testMemory->getData());
    ASSERT_FALSE(nullptr == raw_data_ptr);
    dnnl_memory = dnnl::memory();
    ASSERT_THROW(dnnl_memory = testMemory->getPrimitive(), ov::Exception);
    ASSERT_FALSE(dnnl_memory);
}
