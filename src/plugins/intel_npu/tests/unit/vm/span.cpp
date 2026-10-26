//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_bytecode_utils/span.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <type_traits>

using namespace intel_npu::vm;

namespace {

TEST(VirtualMachineSpanTest, ExposesStdSpanLikeTypeAliases) {
    using IntSpan = Span<int>;

    static_assert(dynamic_extent == static_cast<size_t>(-1));
    static_assert(std::is_same_v<IntSpan::element_type, int>);
    static_assert(std::is_same_v<IntSpan::value_type, int>);
    static_assert(std::is_same_v<IntSpan::size_type, size_t>);
    static_assert(std::is_same_v<IntSpan::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<IntSpan::pointer, int*>);
    static_assert(std::is_same_v<IntSpan::const_pointer, const int*>);
    static_assert(std::is_same_v<IntSpan::reference, int&>);
    static_assert(std::is_same_v<IntSpan::const_reference, const int&>);
    static_assert(IntSpan::extent == dynamic_extent);
}

TEST(VirtualMachineSpanTest, DefaultConstructedSpanIsEmpty) {
    Span<int> span;

    EXPECT_TRUE(span.empty());
    EXPECT_EQ(span.size(), 0u);
    EXPECT_EQ(span.size_bytes(), 0u);
    EXPECT_EQ(span.data(), nullptr);
    EXPECT_EQ(span.begin(), span.end());
}

TEST(VirtualMachineSpanTest, ConstructsFromPointerAndSize) {
    constexpr int32_t value0 = 10;
    constexpr int32_t value1 = 20;
    constexpr int32_t value2 = 30;
    constexpr int32_t value3 = 40;
    std::array<int32_t, 4> values = {value0, value1, value2, value3};

    Span<int32_t> span(values.data(), values.size());

    ASSERT_FALSE(span.empty());
    EXPECT_EQ(span.data(), values.data());
    EXPECT_EQ(span.size(), values.size());
    EXPECT_EQ(span.size_bytes(), values.size() * sizeof(int32_t));
    EXPECT_EQ(span.front(), value0);
    EXPECT_EQ(span.back(), value3);

    const auto iter = std::next(span.begin());
    ASSERT_NE(iter, span.end());
    EXPECT_EQ(*iter, value1);
}

TEST(VirtualMachineSpanTest, ConstructsFromStdArrayMutableAndConst) {
    constexpr int64_t mutable0 = 7;
    constexpr int64_t mutable1 = 8;
    constexpr int64_t mutable2 = 9;
    constexpr int64_t const0 = 11;
    constexpr int64_t const1 = 12;
    std::array<int64_t, 3> mutableValues = {mutable0, mutable1, mutable2};
    const std::array<int64_t, 2> constValues = {const0, const1};

    Span<int64_t> mutableSpan(mutableValues);
    Span<const int64_t> constSpan(constValues);

    ASSERT_EQ(mutableSpan.size(), 3u);
    EXPECT_EQ(mutableSpan.front(), mutable0);
    EXPECT_EQ(mutableSpan.back(), mutable2);

    ASSERT_EQ(constSpan.size(), 2u);
    EXPECT_EQ(constSpan.front(), const0);
    EXPECT_EQ(constSpan.back(), const1);
}

TEST(VirtualMachineSpanTest, ConstructsFromCArray) {
    constexpr int value0 = 1;
    constexpr int value1 = 2;
    constexpr int value2 = 3;
    int values[] = {value0, value1, value2};  // NOLINT(cppcoreguidelines-avoid-c-arrays)

    Span<int> span(values);

    ASSERT_EQ(span.size(), 3u);
    EXPECT_EQ(span.front(), value0);
    EXPECT_EQ(span.back(), value2);
}

TEST(VirtualMachineSpanTest, AllowsConvertingCtorFromMutableToConst) {
    constexpr int firstValue = 3;
    constexpr int lastValue = 5;
    std::array<int, 3> values = {firstValue, 4, lastValue};
    Span<int> mutableSpan(values);

    Span<const int> constSpan(mutableSpan);

    EXPECT_EQ(constSpan.data(), values.data());
    EXPECT_EQ(constSpan.size(), values.size());
    EXPECT_EQ(constSpan.front(), firstValue);
    EXPECT_EQ(constSpan.back(), lastValue);
}

TEST(VirtualMachineSpanTest, SupportsFirstLastAndSubspan) {
    constexpr size_t numValues = 6;
    constexpr int lastValue = 5;
    std::array<int, numValues> values = {0, 1, 2, 3, 4, lastValue};
    Span<int> span(values);

    constexpr size_t firstCount = 2;
    constexpr size_t lastCount = 3;
    constexpr size_t middleOffset = 2;
    constexpr size_t middleCount = 3;
    constexpr size_t tailOffset = 4;

    auto firstTwo = span.first(firstCount);
    auto lastThree = span.last(lastCount);
    auto middle = span.subspan(middleOffset, middleCount);
    auto tail = span.subspan(tailOffset);
    auto middleCt = span.subspan<middleOffset, middleCount>();
    auto tailCt = span.subspan<tailOffset>();

    ASSERT_EQ(firstTwo.size(), 2u);
    EXPECT_EQ(firstTwo.front(), 0);
    EXPECT_EQ(firstTwo.back(), 1);

    ASSERT_EQ(lastThree.size(), 3u);
    EXPECT_EQ(lastThree.front(), 3);
    EXPECT_EQ(lastThree.back(), lastValue);

    ASSERT_EQ(middle.size(), 3u);
    EXPECT_EQ(middle.front(), 2);
    EXPECT_EQ(middle.back(), 4);

    ASSERT_EQ(tail.size(), 2u);
    EXPECT_EQ(tail.front(), 4);
    EXPECT_EQ(tail.back(), lastValue);

    ASSERT_EQ(middleCt.size(), 3u);
    EXPECT_EQ(middleCt.front(), 2);
    EXPECT_EQ(middleCt.back(), 4);

    ASSERT_EQ(tailCt.size(), 2u);
    EXPECT_EQ(tailCt.front(), 4);
    EXPECT_EQ(tailCt.back(), lastValue);
}

TEST(VirtualMachineSpanTest, SupportsZeroLengthSlicingWithoutNullifyingData) {
    std::array<int, 4> values = {1, 2, 3, 4};
    Span<int> span(values);

    const auto firstZero = span.first(0);
    const auto lastZero = span.last(0);
    const auto subspanAtEnd = span.subspan(span.size());
    const auto subspanZeroCount = span.subspan(2, 0);

    EXPECT_TRUE(firstZero.empty());
    EXPECT_EQ(firstZero.data(), values.data());

    EXPECT_TRUE(lastZero.empty());
    EXPECT_EQ(lastZero.data(), std::next(values.data(), static_cast<std::ptrdiff_t>(values.size())));

    EXPECT_TRUE(subspanAtEnd.empty());
    EXPECT_EQ(subspanAtEnd.data(), std::next(values.data(), static_cast<std::ptrdiff_t>(values.size())));

    EXPECT_TRUE(subspanZeroCount.empty());
    EXPECT_EQ(subspanZeroCount.data(), std::next(values.data(), 2));
}

TEST(VirtualMachineSpanTest, IteratorsMatchUnderlyingBufferOrder) {
    constexpr size_t numValues = 5;
    constexpr int value0 = 2;
    constexpr int value1 = 4;
    constexpr int value2 = 6;
    constexpr int value3 = 8;
    constexpr int value4 = 10;
    constexpr int expectedSum = 30;
    std::array<int, numValues> values = {value0, value1, value2, value3, value4};
    Span<int> span(values);

    const int forwardSum = std::accumulate(span.begin(), span.end(), 0);

    int reverseSum = 0;
    for (auto it = span.rbegin(); it != span.rend(); ++it) {
        reverseSum += *it;
    }

    const int constForwardSum = std::accumulate(span.cbegin(), span.cend(), 0);
    int constReverseSum = 0;
    for (auto it = span.crbegin(); it != span.crend(); ++it) {
        constReverseSum += *it;
    }

    EXPECT_EQ(forwardSum, expectedSum);
    EXPECT_EQ(reverseSum, expectedSum);
    EXPECT_EQ(constForwardSum, expectedSum);
    EXPECT_EQ(constReverseSum, expectedSum);
}

TEST(VirtualMachineSpanTest, ExposesByteViewsLikeStdSpan) {
    constexpr uint32_t value0 = 0x11223344U;
    constexpr uint32_t value1 = 0x55667788U;
    std::array<uint32_t, 2> values = {value0, value1};
    Span<uint32_t> span(values);

    const auto readOnlyBytes = as_bytes(span);
    auto writableBytes = as_writable_bytes(span);

    EXPECT_EQ(readOnlyBytes.size(), span.size_bytes());
    EXPECT_EQ(writableBytes.size(), span.size_bytes());
    EXPECT_EQ(readOnlyBytes.data(), static_cast<const std::byte*>(static_cast<const void*>(values.data())));
    EXPECT_EQ(writableBytes.data(), static_cast<std::byte*>(static_cast<void*>(values.data())));
}

}  // namespace
