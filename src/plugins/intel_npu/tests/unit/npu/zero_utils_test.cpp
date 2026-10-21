// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_utils.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

TEST(ZeroUtils, reverseBitsPerByteReversesEveryByte) {
    std::array<uint8_t, 256> data{};
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(i);
    }

    intel_npu::zeroUtils::reverseBitsPerByte(data.data(), data.size());

    for (size_t i = 0; i < data.size(); ++i) {
        uint8_t expected = 0;
        for (size_t bit = 0; bit < 8; ++bit) {
            expected = static_cast<uint8_t>((expected << 1) | ((i >> bit) & 1U));
        }
        EXPECT_EQ(data[i], expected) << "byte " << i;
    }
}

TEST(ZeroUtils, reverseBitsPerByteIsItsOwnInverse) {
    std::vector<uint8_t> data(256);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>((i * 37 + 11) & 0xFF);
    }
    const auto original = data;

    intel_npu::zeroUtils::reverseBitsPerByte(data.data(), data.size());
    intel_npu::zeroUtils::reverseBitsPerByte(data.data(), data.size());

    EXPECT_EQ(data, original);

    std::array<uint8_t, 3> tail{0x01, 0x80, 0xFF};
    intel_npu::zeroUtils::reverseBitsPerByte(tail.data(), tail.size());
    EXPECT_EQ(tail, (std::array<uint8_t, 3>{0x80, 0x01, 0xFF}));
}

TEST(ZeroUtils, isSameStorageSpellingIsSymmetric) {
    using ov::element::Type;
    const std::array<std::pair<Type, Type>, 4> pairs{
        std::pair{ov::element::u8, ov::element::boolean},
        std::pair{ov::element::u1, ov::element::s1},
        std::pair{ov::element::u2, ov::element::i2},
        std::pair{ov::element::u4, ov::element::i4}};

    for (const auto& [first, second] : pairs) {
        EXPECT_TRUE(intel_npu::zeroUtils::isSameStorageSpelling(first, second));
        EXPECT_TRUE(intel_npu::zeroUtils::isSameStorageSpelling(second, first));
    }

    EXPECT_FALSE(intel_npu::zeroUtils::isSameStorageSpelling(ov::element::u2, ov::element::s1));
    EXPECT_FALSE(intel_npu::zeroUtils::isSameStorageSpelling(ov::element::u8, ov::element::i2));
    EXPECT_FALSE(intel_npu::zeroUtils::isSameStorageSpelling(ov::element::u4, ov::element::i2));
    EXPECT_FALSE(intel_npu::zeroUtils::isSameStorageSpelling(ov::element::u1, ov::element::boolean));
    EXPECT_FALSE(intel_npu::zeroUtils::isSameStorageSpelling(ov::element::i4, ov::element::i4));
}

}  // namespace
