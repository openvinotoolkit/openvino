// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <memory>

#include <gtest/gtest.h>

#include <vpu/utils/small_vector.hpp>
#include <vpu/utils/range.hpp>
#include <vpu/utils/handle.hpp>

using namespace testing;

TEST(VPU_SmallVectorTests, SimpleUsage) {
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

TEST(VPU_SmallVectorTests, Equal) {
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

TEST(VPU_SmallVectorTests, Swap) {
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
