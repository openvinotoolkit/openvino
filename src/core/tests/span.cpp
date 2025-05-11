// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/utils/span.hpp"

#include <algorithm>
#include <array>
#include <vector>

#include "gtest/gtest.h"

using namespace ov::reference;

TEST(span_util, create_from_vector) {
    std::vector<int> data{1, 2, 3, 4};
    const auto s = span(data);

    ASSERT_EQ(s.size(), data.size());
    EXPECT_TRUE(std::equal(begin(data), end(data), begin(s)));

    const auto si = span(begin(data), end(data));

    ASSERT_EQ(si.size(), data.size());
    EXPECT_TRUE(std::equal(begin(data), end(data), begin(si)));
}

TEST(span_util, create_from_const_vector) {
    const std::vector<int> data{1, 2, 3, 4};
    const auto s = span(data);

    ASSERT_EQ(s.size(), data.size());
    EXPECT_TRUE(std::equal(begin(data), end(data), begin(s)));

    const auto si = span(begin(data), end(data));

    ASSERT_EQ(si.size(), data.size());
    EXPECT_TRUE(std::equal(begin(data), end(data), begin(si)));
}

TEST(span_util, create_from_memory) {
    std::array<int, 4> data{1, 2, 3, 4};
    const auto s = span(data);

    ASSERT_EQ(s.size(), data.size());
    EXPECT_TRUE(std::equal(begin(data), end(data), begin(s)));
}

TEST(span_util, create_from_const_memory) {
    const std::array<int, 4> data{1, 2, 3, 4};
    const auto s = span(data);

    ASSERT_EQ(s.size(), data.size());
    EXPECT_TRUE(std::equal(begin(data), end(data), begin(s)));
}

TEST(span_util, empty_span_stay_empty_for_drop_front) {
    {
        constexpr std::array<int, 1> data{1};
        auto s = span(data);
        EXPECT_EQ(1, s.size());
        EXPECT_FALSE(s.empty());
        EXPECT_EQ(data.front(), s.front());

        s.drop_front(1);
        EXPECT_EQ(0, s.size());
        EXPECT_TRUE(s.empty());

        s.drop_front(1);
        EXPECT_EQ(0, s.size());
        EXPECT_TRUE(s.empty());
    }
    {
        constexpr std::array<int, 2> data{1, 2};
        auto s = span(data);
        EXPECT_EQ(2, s.size());
        EXPECT_FALSE(s.empty());
        EXPECT_EQ(data.front(), s.front());

        s.drop_front(1);
        EXPECT_FALSE(s.empty());
        EXPECT_EQ(data.back(), s.front());

        s.drop_front(1);
        EXPECT_EQ(0, s.size());
        EXPECT_TRUE(s.empty());

        s.drop_front(1);
        EXPECT_EQ(0, s.size());
        EXPECT_TRUE(s.empty());
    }
}
TEST(span_util, empty_span_stay_empty_for_drop_back) {
    {
        constexpr std::array<int, 1> data{1};
        auto s = span(data);
        EXPECT_EQ(1, s.size());
        EXPECT_FALSE(s.empty());
        EXPECT_EQ(data.front(), s.front());

        s.drop_back(1);
        EXPECT_EQ(0, s.size());
        EXPECT_TRUE(s.empty());

        s.drop_back(1);
        EXPECT_EQ(0, s.size());
        EXPECT_TRUE(s.empty());
    }
    {
        constexpr std::array<int, 2> data{1, 2};
        auto s = span(data);
        EXPECT_EQ(2, s.size());
        EXPECT_FALSE(s.empty());
        EXPECT_EQ(data.back(), s.back());

        s.drop_back(1);
        EXPECT_FALSE(s.empty());
        EXPECT_EQ(data.front(), s.back());

        s.drop_back(1);
        EXPECT_EQ(0, s.size());
        EXPECT_TRUE(s.empty());

        s.drop_back(1);
        EXPECT_EQ(0, s.size());
        EXPECT_TRUE(s.empty());
    }
}

TEST(span_util, create_substring) {
    const std::array<int, 4> data{1, 2, 3, 4};
    const auto s = span(data.data(), data.size());

    {
        const auto sub = s.subspan(1, 1000);
        EXPECT_EQ(sub.size(), data.size() - 1);
        EXPECT_FALSE(sub.empty());
    }
    {
        const auto sub = s.subspan(data.size() - 1);
        EXPECT_EQ(sub.size(), 1);
        EXPECT_FALSE(sub.empty());
    }
    {
        const auto sub = s.subspan(10000, 1000);
        EXPECT_EQ(sub.size(), 0);
        EXPECT_TRUE(sub.empty());
    }
}

TEST(span_util, compare_substr_with_drop_front) {
    const std::array<int, 4> data{1, 2, 3, 4};
    const auto s = span(data.data(), data.size());

    auto sf = s;
    auto ss = s;
    for (size_t i = 0; i != data.size() + 1; ++i) {
        sf.drop_front(1);
        ss = ss.subspan(1);
        EXPECT_EQ(sf.size(), ss.size());
        EXPECT_EQ(sf.empty(), ss.empty());
        if (!sf.empty()) {
            EXPECT_EQ(sf.front(), ss.front());
        }
    }
}

TEST(span_util, drop_elements) {
    const std::array<int, 4> data{1, 2, 3, 4};
    const auto s = span(data.data(), data.size());

    auto length = s.size();
    for (auto sub = s; !sub.empty(); sub.drop_back(1)) {
        EXPECT_EQ(sub.front(), data.front());
        EXPECT_EQ(sub.size(), length);
        length--;
    }

    length = s.size();
    for (auto sub = s; !sub.empty(); sub.drop_front(1)) {
        EXPECT_EQ(sub.back(), data.back());
        EXPECT_EQ(sub.size(), length);
        length--;
    }
}

TEST(span_util, throw_on_out_of_range) {
    std::array<int, 2> data{};
    EXPECT_THROW(Span<char>{}.at(0), std::out_of_range);
    EXPECT_NO_THROW(span(data).at(0));
    EXPECT_NO_THROW(span(data).at(1));
    EXPECT_THROW(span(data).at(2), std::out_of_range);
    EXPECT_THROW(span(data).at(3), std::out_of_range);
}
