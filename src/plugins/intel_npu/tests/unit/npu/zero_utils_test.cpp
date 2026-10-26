// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_utils.hpp"

#include <gtest/gtest.h>

#include <array>
#include <utility>

namespace {

TEST(ZeroUtils, reverseBitsPerByteReversesEveryByte) {
    // The direction, pinned by hand: the low bit becomes the high one. The sweep below recomputes the table's
    // own loop and would pass with every byte mapped the same wrong way; these four literals would not.
    const std::array<std::pair<uint8_t, uint8_t>, 4> known{std::pair{0x01, 0x80},
                                                           std::pair{0x0F, 0xF0},
                                                           std::pair{0xAA, 0x55},
                                                           std::pair{0x80, 0x01}};
    for (const auto& [stored, expected] : known) {
        auto byte = stored;
        intel_npu::zeroUtils::reverseBitsPerByte(&byte, 1);
        EXPECT_EQ(byte, expected) << std::hex << static_cast<int>(stored);
    }

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

    // Only the supplied bytes are touched, including an odd-length payload.
    std::array<uint8_t, 3> tail{0x01, 0x80, 0xFF};
    intel_npu::zeroUtils::reverseBitsPerByte(tail.data(), tail.size());
    EXPECT_EQ(tail, (std::array<uint8_t, 3>{0x80, 0x01, 0xFF}));
}

// The rule every write into a device tensor applies: a `u1` payload is reversed byte by byte, because the device
// indexes it LSB-first, and every other storage is copied as it is.
TEST(ZeroUtils, copyPayloadInDeviceBitOrderReversesOnlyAOneBitPayload) {
    const std::array<uint8_t, 3> stored{0x01, 0x0F, 0xAA};

    std::array<uint8_t, 3> reversed{};
    intel_npu::zeroUtils::copyPayloadInDeviceBitOrder(ov::element::u1, reversed.data(), stored.data(), stored.size());
    EXPECT_EQ(reversed, (std::array<uint8_t, 3>{0x80, 0xF0, 0x55}));

    std::array<uint8_t, 3> wider{};
    intel_npu::zeroUtils::copyPayloadInDeviceBitOrder(ov::element::i4, wider.data(), stored.data(), stored.size());
    EXPECT_EQ(wider, stored);
}

TEST(ZeroUtils, isSameStorageSpellingAcceptsI2AndU2InBothOrders) {
    EXPECT_TRUE(intel_npu::zeroUtils::isSameStorageSpelling(ov::element::u2, ov::element::i2));
    EXPECT_TRUE(intel_npu::zeroUtils::isSameStorageSpelling(ov::element::i2, ov::element::u2));
    EXPECT_FALSE(intel_npu::zeroUtils::isSameStorageSpelling(ov::element::u4, ov::element::i4));
    EXPECT_FALSE(intel_npu::zeroUtils::isSameStorageSpelling(ov::element::u2, ov::element::u4));
}

}  // namespace
