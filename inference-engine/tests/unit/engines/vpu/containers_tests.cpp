// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <memory>

#include <gtest/gtest.h>

#include <vpu/utils/containers.hpp>
#include <vpu/utils/range.hpp>
#include <vpu/utils/handle.hpp>

using namespace testing;

namespace {

struct TestStruct final : public vpu::EnableHandleFromThis<TestStruct> {
    int val = 0;
    vpu::IntrusivePtrListNode<TestStruct> node1;
    vpu::IntrusivePtrListNode<TestStruct> node2;
    explicit TestStruct(int val) : val(val), node1(this), node2(this) {}
};

}

TEST(VPU_Containers, SmallVector_API) {
    std::vector<int> vec1;
    vpu::SmallVector<int, 5> vec2;

    for (size_t i = 0; i < 5; ++i) {
        vec1.push_back(i);
        vec2.push_back(i);
    }

    for (size_t i = 0; i < vec1.size(); ++i) {
        ASSERT_EQ(vec1.at(i), vec2.at(i));
    }

    vec1.clear();
    vec2.clear();

    for (size_t i = 0; i < 5; ++i) {
        vec1.push_back(i);
    }
    vec2.insert(vec2.end(), vec1.begin(), vec1.end());

    auto it1 = std::find(vec1.begin(), vec1.end(), 2);
    auto it2 = std::find(vec2.begin(), vec2.end(), 2);

    ASSERT_NE(it1, vec1.end());
    ASSERT_NE(it2, vec2.end());

    vec1.erase(it1);
    vec2.erase(it2);

    for (size_t i = 0; i < vec1.size(); ++i) {
        ASSERT_EQ(vec1.at(i), vec2.at(i));
    }

    vec1.push_back(15);
    vec1.push_back(16);

    vec2.push_back(15);
    vec2.push_back(16);

    for (size_t i = 0; i < vec1.size(); ++i) {
        ASSERT_EQ(vec1.at(i), vec2.at(i));
    }
}

TEST(VPU_Containers, SmallVector_Equal) {
    vpu::SmallVector<int, 5> vec1;
    vpu::SmallVector<int, 5> vec2;
    vpu::SmallVector<int, 5> vec3;

    for (size_t i = 0; i < 5; ++i) {
        vec1.push_back(i);
        vec2.push_back(i);
        vec3.push_back(i + 1);
    }

    ASSERT_EQ(vec1, vec2);
    ASSERT_NE(vec1, vec3);
}

TEST(VPU_Containers, SmallVector_Swap) {
    vpu::SmallVector<int, 5> vec1;
    vpu::SmallVector<int, 5> vec2;

    for (size_t i = 0; i < 5; ++i) {
        vec1.push_back(i);
        vec2.push_back(5 - i);
    }

    vec1.swap(vec2);

    for (size_t i = 0; i < 5; ++i) {
        ASSERT_EQ(vec1[i], 5 - i);
        ASSERT_EQ(vec2[i], i);
    }
}

template <class Cont>
Cont buildTestVector(int contSize) {
    Cont vec;

    for (int i = 0; i < contSize; ++i) {
        vec.push_back(i);
    }

    return vec;
}

TEST(VPU_Containers, IntrusivePtrList) {
    const int count = 5;
    int gold = 0;

    std::vector<std::shared_ptr<TestStruct>> base;
    for (int i = 0; i < count; ++i) {
        base.push_back(std::make_shared<TestStruct>(i));
    }

    vpu::IntrusivePtrList<TestStruct> list1(&TestStruct::node1);
    vpu::IntrusivePtrList<TestStruct> list2(&TestStruct::node2);

    for (int i = 0; i < count; ++i) {
        list1.push_back(base[i]);
    }

    ASSERT_FALSE(list1.empty());
    ASSERT_TRUE(list2.empty());

    gold = 0;
    for (const auto& ptr1 : list1) {
        ASSERT_NE(ptr1, nullptr);
        ASSERT_EQ(ptr1->val, gold);
        ASSERT_EQ(ptr1.get(), base[ptr1->val].get());
        ++gold;
    }
    ASSERT_EQ(gold, count);

    for (int i = 0; i < count / 2; ++i) {
        list2.push_back(base[i]);
    }

    ASSERT_FALSE(list2.empty());

    gold = 0;
    for (const auto& ptr2 : list2) {
        ASSERT_NE(ptr2, nullptr);
        ASSERT_EQ(ptr2->val, gold);
        ASSERT_EQ(ptr2.get(), base[ptr2->val].get());

        list1.erase(ptr2);

        ++gold;
    }
    ASSERT_EQ(gold, count / 2);

    gold = count / 2;
    for (const auto& ptr1 : list1) {
        ASSERT_NE(ptr1, nullptr);
        ASSERT_EQ(ptr1->val, gold);
        ASSERT_EQ(ptr1.get(), base[ptr1->val].get());
        ++gold;
    }
    ASSERT_EQ(gold, count);
}

TEST(VPU_Containers, IntrusivePtrList_MoveFromOneListToAnother) {
    const int count = 5;

    std::list<std::shared_ptr<TestStruct>> base;

    vpu::IntrusivePtrList<TestStruct> list1(&TestStruct::node1);
    vpu::IntrusivePtrList<TestStruct> list2(&TestStruct::node1);

    for (int i = 0; i < count; ++i) {
        auto ptr = std::make_shared<TestStruct>(i);
        base.push_back(ptr);
        list1.push_back(ptr);
    }

    ASSERT_EQ(list1.size(), base.size());
    ASSERT_TRUE(list2.empty());

    for (const auto& item : list1) {
        list1.erase(item);
        list2.push_back(item);
    }

    ASSERT_TRUE(list1.empty());
    ASSERT_EQ(list2.size(), base.size());
}

TEST(VPU_Containers, IntrusivePtrList_ReleaseOrigObject) {
    const int count = 5;
    int gold = 0;

    std::list<std::shared_ptr<TestStruct>> base;

    vpu::IntrusivePtrList<TestStruct> list(&TestStruct::node1);

    for (int i = 0; i < count; ++i) {
        auto ptr = std::make_shared<TestStruct>(i);
        base.push_back(ptr);
        list.push_back(ptr);
    }

    ASSERT_EQ(list.size(), base.size());

    base.pop_front();
    ASSERT_EQ(list.size(), base.size());

    base.pop_back();
    ASSERT_EQ(list.size(), base.size());

    list.clear();
    ASSERT_TRUE(list.empty());

    gold = 0;
    for (const auto& item : base) {
        ASSERT_EQ(item->val, gold + 1);
        ++gold;
    }
    ASSERT_EQ(gold, count - 2);
}
