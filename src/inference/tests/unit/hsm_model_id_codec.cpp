// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_model_id_codec.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "openvino/runtime/hsm_format.hpp"

namespace ov::test {
namespace hsm = ov::runtime::hsm;
namespace {

using WireId = std::array<uint8_t, sizeof(uint64_t)>;

constexpr uint64_t sample_id = 0x0123456789ABCDEF;
constexpr WireId sample_wire{0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01};

constexpr uint64_t max_id = std::numeric_limits<uint64_t>::max();
constexpr WireId max_wire{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

ov::util::MemoryView view_of(const WireId& wire) {
    return {reinterpret_cast<const std::byte*>(wire.data()), wire.size()};
}

void expect_wraps_to(const uint64_t& id, const WireId& expected) {
    const auto wrapped = hsm::wrap_model_id(id);
    ASSERT_EQ(wrapped.size(), expected.size());
    EXPECT_EQ(std::memcmp(wrapped.data(), expected.data(), expected.size()), 0);
}

}  // namespace

TEST(HsmModelIdSectionTest, wrap_emits_the_little_endian_wire_bytes) {
    expect_wraps_to(sample_id, sample_wire);
    expect_wraps_to(max_id, max_wire);
    expect_wraps_to(0, WireId{});
}

TEST(HsmModelIdSectionTest, unwrap_reads_the_little_endian_wire_bytes) {
    EXPECT_EQ(hsm::unwrap_model_id(view_of(sample_wire)), sample_id);
    EXPECT_EQ(hsm::unwrap_model_id(view_of(max_wire)), max_id);
    EXPECT_EQ(hsm::unwrap_model_id(view_of(WireId{})), 0u);
}

TEST(HsmModelIdSectionTest, unwrap_ignores_the_zero_fill_of_a_full_inline_slot) {
    // An inline payload reads back as the whole slot, zero-filled past the bytes actually written.
    std::array<uint8_t, sizeof(hsm::ManifestEntry{}.inline_bytes)> slot{};
    std::copy(sample_wire.begin(), sample_wire.end(), slot.begin());

    EXPECT_EQ(hsm::unwrap_model_id({reinterpret_cast<const std::byte*>(slot.data()), slot.size()}), sample_id);
}

TEST(HsmModelIdSectionTest, unwrap_rejects_a_payload_too_short_to_hold_an_id) {
    const std::array<std::byte, sizeof(uint64_t) - 1> too_short{};

    EXPECT_FALSE(hsm::unwrap_model_id({too_short.data(), too_short.size()}).has_value());
    EXPECT_FALSE(hsm::unwrap_model_id({}).has_value());
}

}  // namespace ov::test
