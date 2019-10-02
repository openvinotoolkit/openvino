// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <list>

#include <gtest/gtest.h>

#include <vpu/utils/cache.hpp>

using namespace testing;

TEST(VPU_CacheTests, Value) {
    int gold = 0;

    vpu::ValueCache<int> cache([&]() {
        return gold;
    });

    ASSERT_EQ(gold, cache.get()) << "First usage";

    gold = 1;
    cache.reset();

    ASSERT_EQ(gold, cache.get()) << "After reset";
}

TEST(VPU_CacheTests, ContainerRange) {
    int count = 3;
    int gold = 0;

    vpu::ContainerRangeCache<std::vector<int>> cache;

    cache.setInitializer([&]() {
        std::vector<int> vec;
        for (int i = 0; i < count; ++i) {
            vec.push_back(i);
        }
        return vec;
    });

    gold = 0;
    for (auto val : cache.get()) {
        ASSERT_EQ(val, gold);
        gold++;
    }
    ASSERT_EQ(gold, count) << "First usage";

    count = 5;
    cache.reset();

    gold = 0;
    for (auto val : cache.get()) {
        ASSERT_EQ(val, gold);
        gold++;
    }
    ASSERT_EQ(gold, count) << "After reset";

    count = 7;
    cache.setInitializer([&]() {
        std::list<int> list;
        for (int i = 0; i < count; ++i) {
            list.push_back(i);
        }
        return std::move(list) | vpu::asRange();
    });

    gold = 0;
    for (auto val : cache.get()) {
        ASSERT_EQ(val, gold);
        gold++;
    }
    ASSERT_EQ(gold, count) << "Convert container";
}
