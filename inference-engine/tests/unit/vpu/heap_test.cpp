// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "vpu/utils/heap.hpp"


template<typename T>
std::vector<T> MakeHeap(const std::initializer_list<T> &list) {
    std::vector<T> v;
    v.reserve(list.size());
    for (auto i : list) {
        v.push_back(i);
        std::push_heap(v.begin(), v.end());
    }
    return v;
}

template<typename T>
std::vector<T> FixedMaxHeapToVector(const vpu::FixedMaxHeap<T> &FMH) {
    std::vector<T> v;
    for (auto i : FMH)
        v.push_back(i);
    return v;
}

TEST(VPU_FixedMaxHeapTest, DefaultConstructor) {
    ASSERT_NO_THROW(vpu::FixedMaxHeap<int>(10));
}

TEST(VPU_FixedMaxHeapTest, ConstructorWithInit) {
    std::initializer_list<int> arr = {3, 4, 1, 2, 5, 9, 5};
    std::vector<int> ref_v = MakeHeap(arr);

    vpu::FixedMaxHeap<int> heap(arr.size(), arr);
    std::vector<int> v = FixedMaxHeapToVector(heap);

    ASSERT_THAT(v, testing::ElementsAreArray(ref_v));
}


TEST(VPU_FixedMaxHeapTest, CanBeCopied) {
    std::initializer_list<int> arr = {3, 4, 1, 2, 5, 9, 5};
    std::vector<int> ref_v = MakeHeap(arr);

    {
        vpu::FixedMaxHeap<int> heap(arr.size(), arr);
        vpu::FixedMaxHeap<int> heap2(heap);
        std::vector<int> v = FixedMaxHeapToVector(heap2);

        EXPECT_THAT(v, testing::ElementsAreArray(ref_v));
    }

    {
        vpu::FixedMaxHeap<int> heap(arr.size(), arr);
        vpu::FixedMaxHeap<int> heap2(arr.size());
        heap2 = heap;
        std::vector<int> v = FixedMaxHeapToVector(heap2);

        EXPECT_THAT(v, testing::ElementsAreArray(ref_v));
    }

    // TODO: extend with rvalue copy
}


TEST(VPU_FixedMaxHeapTest, Front) {
    {
        vpu::FixedMaxHeap<int> heap(3, {3, 4, 100});
        EXPECT_EQ(heap.front(), 100);
    }

    {
        vpu::FixedMaxHeap<int> heap(3, {0, -4, -100});
        EXPECT_EQ(heap.front(), 0);
    }
}

TEST(VPU_FixedMaxHeapTest, Size) {
    {
        vpu::FixedMaxHeap<int> heap(0);
        EXPECT_EQ(heap.size(), 0);
    }

    {
        vpu::FixedMaxHeap<int> heap(3);
        EXPECT_EQ(heap.size(), 0);
    }

    {
        vpu::FixedMaxHeap<int> heap(3, {3, 4, 100});
        EXPECT_EQ(heap.size(), 3);
    }

    {
        vpu::FixedMaxHeap<int> heap(3, {0, -4, -100, 10, 15});
        EXPECT_EQ(heap.size(), 3);
    }
}

TEST(VPU_FixedMaxHeapTest, Empty) {
    {
        vpu::FixedMaxHeap<int> heap(3);
        EXPECT_TRUE(heap.empty());
    }

    {
        vpu::FixedMaxHeap<int> heap(3, {3, 4, 100});
        EXPECT_FALSE(heap.empty());
    }
}

TEST(VPU_FixedMaxHeapTest, Push) {
    {
        std::vector<int> ref_v = MakeHeap({3, 4, 5, 100});
        vpu::FixedMaxHeap<int> heap(4, {3, 4, 5});
        heap.push(100);
        std::vector<int> v = FixedMaxHeapToVector(heap);
        EXPECT_THAT(v, testing::ElementsAreArray(ref_v));
    }

    {
        std::vector<int> ref_v = MakeHeap({3, 4, 5});
        vpu::FixedMaxHeap<int> heap(3, {3, 4, 5});
        heap.push(100);
        std::vector<int> v = FixedMaxHeapToVector(heap);
        EXPECT_THAT(v, testing::ElementsAreArray(ref_v));
    }

    {
        std::vector<int> ref_v = MakeHeap({3, 4, 2});
        vpu::FixedMaxHeap<int> heap(3, {3, 4, 5});
        heap.push(2);
        std::vector<int> v = FixedMaxHeapToVector(heap);
        EXPECT_THAT(v, testing::ElementsAreArray(ref_v));
    }
}

TEST(VPU_FixedMaxHeapTest, Sorted) {
    vpu::FixedMaxHeap<int> heap(10, {10, -9, 8, -7, 6, -5, 4, -3, 2, -1});
    auto s = heap.sorted();
    ASSERT_FALSE(s.empty());
    ASSERT_TRUE(std::is_sorted(s.begin(), s.end()));
}

TEST(VPU_FixedMaxHeapTest, Print) {
    std::initializer_list<int> arr = {10, -9, 8, -7, 6, -5, 4, -3, 2, -1};
    int heap_capacity = arr.size() + 2;
    vpu::FixedMaxHeap<int> heap(heap_capacity, arr);

    std::ostringstream ref_ostr;
    ref_ostr << "Heap [" << arr.size() << " / " << heap_capacity << "]: ";
    for (auto i : heap)
        ref_ostr << i << " ";
    std::string ref_s = ref_ostr.str();

    std::ostringstream ostr;
    ostr << heap;
    std::string s = ostr.str();
    ASSERT_STREQ(ref_s.c_str(), s.c_str());
}

TEST(VPU_FixedMaxHeapTest, FixedMaxHeapFloat) {
    vpu::FixedMaxHeap<float> heap(3, {3., 4., 5.});

    for (auto i : heap)
        EXPECT_EQ(typeid(i), typeid(float));

    EXPECT_NO_THROW(heap.push(0.));

    for (auto i : heap.sorted())
        EXPECT_EQ(typeid(i), typeid(float));
}