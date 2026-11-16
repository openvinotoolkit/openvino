// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <atomic>
#include <thread>

#include "cpu_memory.h"
#include "dnnl_scratch_pad.h"
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

namespace {
MemoryDescPtr bytesDesc(size_t size) {
    return std::make_shared<CpuBlockedMemoryDesc>(ov::element::u8, Shape{size});
}
}  // namespace

TEST(ScratchPadTest, MemoryAtOffsetGrowsTheSharedBlock) {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    DnnlScratchPad scratchPad(eng);

    auto mem = scratchPad.createScratchPadMem(bytesDesc(100), 256);
    ASSERT_GE(scratchPad.size(), 256U + 100U);

    // The memory starts `offset` bytes after the memory that starts the block.
    auto first = scratchPad.createScratchPadMem(bytesDesc(64));
    ASSERT_EQ(static_cast<uint8_t*>(first->getData()) + 256, static_cast<uint8_t*>(mem->getData()));
}

TEST(ScratchPadTest, OffsetDoesNotDependOnTheMemorySize) {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    DnnlScratchPad scratchPad(eng);

    auto first = scratchPad.createScratchPadMem(bytesDesc(8));
    for (const size_t size : {1U, 7U, 64U, 1000U, 4096U}) {
        auto mem = scratchPad.createScratchPadMem(bytesDesc(size), 192);
        ASSERT_EQ(static_cast<uint8_t*>(first->getData()) + 192, static_cast<uint8_t*>(mem->getData())) << size;
    }
}

TEST(ScratchPadTest, ViewStartsBehindItsOffsetAndNests) {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    DnnlScratchPad scratchPad(eng);
    auto first = scratchPad.createScratchPadMem(bytesDesc(8));

    DnnlScratchPad view(scratchPad, 128);
    DnnlScratchPad nested(view, 64);
    auto viewMem = view.createScratchPadMem(bytesDesc(32));
    auto nestedMem = nested.createScratchPadMem(bytesDesc(32));
    auto nestedMemAt = nested.createScratchPadMem(bytesDesc(32), 16);

    const auto* base = static_cast<uint8_t*>(first->getData());
    ASSERT_EQ(base + 128, static_cast<uint8_t*>(viewMem->getData()));
    ASSERT_EQ(base + 192, static_cast<uint8_t*>(nestedMem->getData()));
    ASSERT_EQ(base + 208, static_cast<uint8_t*>(nestedMemAt->getData()));
    ASSERT_GE(scratchPad.size(), 208U + 32U);
}

TEST(ScratchPadTest, MemoryAtOffsetFollowsReallocation) {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    DnnlScratchPad scratchPad(eng);

    auto first = scratchPad.createScratchPadMem(bytesDesc(16));
    auto mem = scratchPad.createScratchPadMem(bytesDesc(64), 64);
    // The primitive is created before the block grows: its data handle must move with the block.
    auto prim = mem->getPrimitive();
    ASSERT_EQ(static_cast<uint8_t*>(first->getData()) + 64, prim.get_data_handle());

    auto big = scratchPad.createScratchPadMem(bytesDesc(1U << 20));
    ASSERT_EQ(static_cast<uint8_t*>(first->getData()) + 64, static_cast<uint8_t*>(mem->getData()));
    ASSERT_EQ(mem->getData(), prim.get_data_handle());
}

TEST(ScratchPadTest, EmptyMemoryAtOffsetDoesNotGrowTheBlock) {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    DnnlScratchPad scratchPad(eng);

    ASSERT_NO_THROW(scratchPad.createScratchPadMem(bytesDesc(0), 128));
    ASSERT_EQ(scratchPad.size(), 0U);
}
