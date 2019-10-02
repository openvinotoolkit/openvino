// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/utils/heap.hpp>

#include <gtest/gtest.h>

using namespace vpu;
using namespace testing;

TEST(VPU_FixedMaxHeapTest, EmptyHeap) {
    FixedMaxHeap<int> heap(10);
    auto s = heap.sorted();
    ASSERT_TRUE(s.empty());
}

TEST(VPU_FixedMaxHeapTest, HeapSizeOne) {
    FixedMaxHeap<int> heap(10);
    heap.push(5);
    auto s = heap.sorted();
    ASSERT_EQ(s.size(), 1);
    ASSERT_EQ(s[0], 5);
}

TEST(VPU_FixedMaxHeapTest, HeapSizeZero) {
    FixedMaxHeap<int> heap(0);
    heap.push(5);
    auto s = heap.sorted();
    ASSERT_TRUE(s.empty());
}

TEST(VPU_FixedMaxHeapTest, HeapSizeAtCapacity) {
    FixedMaxHeap<int> heap(10);
    for (int i = 10; i > 0; --i) {
        heap.push(i);
    }
    auto s = heap.sorted();
    ASSERT_TRUE(!s.empty());
    ASSERT_TRUE(std::is_sorted(s.begin(), s.end()));
}

TEST(VPU_FixedMaxHeapTest, HeapHasAllElementsPushedInDescendingOrder) {
    FixedMaxHeap<int> heap(10);
    for (int i = 10; i > 0; --i) {
        heap.push(i);
    }
    auto s = heap.sorted();
    ASSERT_TRUE(s.size() == 10);
    ASSERT_TRUE(std::is_sorted(s.begin(), s.end()));
}

TEST(VPU_FixedMaxHeapTest, HeapHasAllElementsPushedInAscendingOrder) {
    FixedMaxHeap<int> heap(10);
    for (int i = 0; i < 10; ++i) {
        heap.push(i);
    }
    auto s = heap.sorted();
    ASSERT_TRUE(s.size() == 10);
    ASSERT_TRUE(std::is_sorted(s.begin(), s.end()));
}

TEST(VPU_FixedMaxHeapTest, HeapSizeAboveCapacity) {
    FixedMaxHeap<int> heap(10);
    for (int i = 15; i > 0; --i) {
        heap.push(i);
    }
    auto s = heap.sorted();
    ASSERT_TRUE(!s.empty());
    ASSERT_TRUE(std::is_sorted(s.begin(), s.end()));
}

TEST(VPU_FixedMaxHeapTest, Sorted) {
    FixedMaxHeap<int> heap(10);
    for (int i = 15; i > 0; --i) {
        heap.push(i+1);
        heap.push(i-1);
        heap.push(i);
    }
    auto s = heap.sorted();
    ASSERT_TRUE(!s.empty());
    ASSERT_TRUE(std::is_sorted(s.begin(), s.end()));
}

TEST(VPU_FixedMaxHeapTest, Print) {
    FixedMaxHeap<int> heap(10);
    std::ostringstream ostr;
    for (int i = 10; i > 0; --i) {
        heap.push(i);
    }
    ostr << heap;
    std::string s = ostr.str();
    ASSERT_TRUE(!s.empty());
}



