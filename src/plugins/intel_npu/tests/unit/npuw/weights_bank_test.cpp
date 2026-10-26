// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weights_bank.hpp"

#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "openvino/runtime/make_tensor.hpp"

namespace {

// The bank's one-shot fill is the writer for the allocations the plugin's copy path never touches, so it owns the
// device's bit order: a `u1` payload is stored MSB-first and the device reads it LSB-first.
TEST(WeightsBank, FillDeviceTensorReversesAOneBitPayload) {
    const std::array<uint8_t, 4> stored{0x01, 0x0F, 0xAA, 0x80};
    ov::Tensor src(ov::element::u1, ov::Shape{32}, const_cast<uint8_t*>(stored.data()));
    ov::Tensor dst(ov::element::u1, ov::Shape{32});
    ASSERT_EQ(dst.get_byte_size(), stored.size());

    ov::npuw::weights::fill_device_tensor(src, dst);

    const auto* actual = static_cast<const uint8_t*>(dst.data());
    const std::array<uint8_t, 4> expected{0x80, 0xF0, 0x55, 0x01};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(actual[i], expected[i]) << "byte " << i;
    }
}

// Every other storage is already in the order the device reads, so the fill is the plain copy.
TEST(WeightsBank, FillDeviceTensorCopiesAWiderPayloadAsIs) {
    const std::array<uint8_t, 4> stored{0xE4, 0x1D, 0xA5, 0x3C};
    ov::Tensor src(ov::element::i4, ov::Shape{8}, const_cast<uint8_t*>(stored.data()));
    ov::Tensor dst(ov::element::i4, ov::Shape{8});
    ASSERT_EQ(dst.get_byte_size(), stored.size());

    ov::npuw::weights::fill_device_tensor(src, dst);

    const auto* actual = static_cast<const uint8_t*>(dst.data());
    for (std::size_t i = 0; i < stored.size(); ++i) {
        EXPECT_EQ(actual[i], stored[i]) << "byte " << i;
    }
}

}  // namespace
