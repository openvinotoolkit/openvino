// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <memory>

#include <gtest/gtest.h>

#include <vpu/utils/intrusive_handle_list.hpp>
#include <vpu/utils/range.hpp>
#include <vpu/utils/handle.hpp>

using namespace testing;

namespace {

struct TestStruct final : public vpu::EnableHandle {
    int val = 0;
    vpu::IntrusiveHandleListNode<TestStruct> node1;
    vpu::IntrusiveHandleListNode<TestStruct> node2;
    explicit TestStruct(int val) : val(val), node1(this), node2(this) {}
};

}

TEST(VPU_IntrusiveHandleListTest, SimpleUsage) {
    const int count = 5;
    int gold = 0;

    std::vector<std::shared_ptr<TestStruct>> base;
    for (int i = 0; i < count; ++i) {
        base.push_back(std::make_shared<TestStruct>(i));
    }

    vpu::IntrusiveHandleList<TestStruct> list1(&TestStruct::node1);
    vpu::IntrusiveHandleList<TestStruct> list2(&TestStruct::node2);

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

TEST(VPU_IntrusiveHandleListTest, MoveFromOneListToAnother) {
    const int count = 5;

    std::list<std::shared_ptr<TestStruct>> base;

    vpu::IntrusiveHandleList<TestStruct> list1(&TestStruct::node1);
    vpu::IntrusiveHandleList<TestStruct> list2(&TestStruct::node1);

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

TEST(VPU_IntrusiveHandleListTest, ReleaseOrigObject) {
    const int count = 5;
    int gold = 0;

    std::list<std::shared_ptr<TestStruct>> base;

    vpu::IntrusiveHandleList<TestStruct> list(&TestStruct::node1);

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

TEST(VPU_IntrusiveHandleListTest, ReverseIter) {
    const int count = 5;
    int gold = 0;

    std::vector<std::shared_ptr<TestStruct>> base;
    for (int i = 0; i < count; ++i) {
        base.push_back(std::make_shared<TestStruct>(i));
    }

    vpu::IntrusiveHandleList<TestStruct> list(&TestStruct::node1);

    for (int i = 0; i < count; ++i) {
        list.push_back(base[i]);
    }

    gold = count - 1;
    for (auto it = list.rbegin(); it != list.rend(); ++it) {
        auto ptr = *it;
        ASSERT_NE(ptr, nullptr);
        ASSERT_EQ(ptr->val, gold);
        ASSERT_EQ(ptr.get(), base[ptr->val].get());
        --gold;
    }
    ASSERT_EQ(gold, -1);
}

TEST(VPU_IntrusiveHandleListTest, IteratorCopyAndMove) {
    const int count = 5;
    int gold = 0;

    std::vector<std::shared_ptr<TestStruct>> base;
    for (int i = 0; i < count; ++i) {
        base.push_back(std::make_shared<TestStruct>(i));
    }

    vpu::IntrusiveHandleList<TestStruct> list(&TestStruct::node1);

    for (int i = 0; i < count; ++i) {
        list.push_back(base[i]);
    }

    ASSERT_FALSE(list.empty());

    gold = 0;
    for (auto it1 = list.begin(); it1 != list.end(); ++it1) {
        ASSERT_EQ((*it1)->val, gold);

        auto it2 = it1; // Copy
        ASSERT_EQ((*it2)->val, gold);

        auto it3 = std::move(it2);
        ASSERT_EQ((*it3)->val, gold);
        ASSERT_EQ(it2, list.end());

        ++gold;
    }
    ASSERT_EQ(gold, count);
}


TEST(VPU_IntrusiveHandleListTest, Move) {
    const int count = 5;
    int gold = 0;

    std::list<std::shared_ptr<TestStruct>> base;

    vpu::IntrusiveHandleList<TestStruct> list1(&TestStruct::node1);
    vpu::IntrusiveHandleList<TestStruct> list2(&TestStruct::node1);

    for (int i = 0; i < count; ++i) {
        auto ptr = std::make_shared<TestStruct>(i);
        base.push_back(ptr);
        list1.push_back(ptr);
    }

    ASSERT_EQ(list1.size(), count);
    ASSERT_TRUE(list2.empty());

    list2 = std::move(list1);

    ASSERT_EQ(list2.size(), count);
    ASSERT_TRUE(list1.empty());

    gold = 0;
    for (const auto& item : list2) {
        ASSERT_EQ(item->val, gold);
        ++gold;
    }
    ASSERT_EQ(gold, count);
}
