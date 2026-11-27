// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

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

TEST(ScratchPadTest, ChildMemoryOverlapsNeitherItsParentNorItsSiblings) {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    DnnlScratchPad scratchPad(eng);

    auto parentMem = scratchPad.createScratchPadMem(bytesDesc(256));
    auto firstMem = scratchPad.child(0)->createScratchPadMem(bytesDesc(256));
    auto secondMem = scratchPad.child(1)->createScratchPadMem(bytesDesc(256));
    auto nestedMem = scratchPad.child(0)->child(0)->createScratchPadMem(bytesDesc(256));

    const std::vector<uint8_t*> starts{static_cast<uint8_t*>(parentMem->getData()),
                                       static_cast<uint8_t*>(firstMem->getData()),
                                       static_cast<uint8_t*>(secondMem->getData()),
                                       static_cast<uint8_t*>(nestedMem->getData())};
    for (size_t i = 0; i < starts.size(); ++i) {
        for (size_t j = i + 1; j < starts.size(); ++j) {
            ASSERT_TRUE(starts[i] + 256 <= starts[j] || starts[j] + 256 <= starts[i]) << i << " and " << j;
        }
    }
}

TEST(ScratchPadTest, ChildIndexAlwaysReturnsTheSamePad) {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    DnnlScratchPad scratchPad(eng);

    // A sparse index first: the gaps stay empty and the later indices are distinct pads.
    auto far = scratchPad.child(3);
    auto near = scratchPad.child(0);
    ASSERT_NE(far, near);
    ASSERT_EQ(far, scratchPad.child(3));
    ASSERT_EQ(near, scratchPad.child(0));
    ASSERT_NE(scratchPad.child(1), near);
    // Indices are local to the pad: the child of a child is not the child of the parent.
    ASSERT_NE(near->child(0), near);
    ASSERT_NE(near->child(0), scratchPad.child(0)->child(1));
}

TEST(ScratchPadTest, ReallocatingOnePadKeepsTheDataOfTheOthers) {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    DnnlScratchPad scratchPad(eng);

    auto parentMem = scratchPad.createScratchPadMem(bytesDesc(64));
    auto childMem = scratchPad.child(0)->createScratchPadMem(bytesDesc(64));
    std::fill_n(static_cast<uint8_t*>(childMem->getData()), 64, uint8_t{0xAB});
    // The primitive is created before the pads grow: its data handle must follow its own block.
    auto prim = childMem->getPrimitive();

    // The parent grows: the child is untouched.
    auto bigParent = scratchPad.createScratchPadMem(bytesDesc(1U << 20));
    for (size_t i = 0; i < 64; ++i) {
        ASSERT_EQ(static_cast<uint8_t*>(childMem->getData())[i], 0xAB) << i;
    }

    // The child grows: the memory and its primitive follow, the parent keeps its block.
    auto* parentData = parentMem->getData();
    auto bigChild = scratchPad.child(0)->createScratchPadMem(bytesDesc(1U << 20));
    ASSERT_EQ(parentMem->getData(), parentData);
    ASSERT_EQ(childMem->getData(), prim.get_data_handle());
}

TEST(ScratchPadTest, SizeCountsTheChildrenRecursively) {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    DnnlScratchPad scratchPad(eng);
    ASSERT_EQ(scratchPad.size(), 0U);

    auto parentMem = scratchPad.createScratchPadMem(bytesDesc(100));
    const size_t parentOnly = scratchPad.size();
    ASSERT_GE(parentOnly, 100U);

    auto childMem = scratchPad.child(2)->createScratchPadMem(bytesDesc(200));
    auto nestedMem = scratchPad.child(2)->child(1)->createScratchPadMem(bytesDesc(300));
    ASSERT_GE(scratchPad.size(), parentOnly + 200U + 300U);
}
