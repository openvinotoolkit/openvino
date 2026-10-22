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
#include <type_traits>

#include "openvino/runtime/hsm_format.hpp"
#include "openvino/runtime/hsm_reader.hpp"

namespace ov::test {
namespace hsm = ov::runtime::hsm;
namespace {

using WireId = std::array<uint8_t, sizeof(uint64_t)>;

constexpr uint64_t sample_id = 0x0123456789ABCDEF;
constexpr WireId sample_wire{0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01};

constexpr uint64_t max_id = std::numeric_limits<uint64_t>::max();
constexpr WireId max_wire{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

static_assert(std::is_convertible_v<decltype(&hsm::decode_model_id), hsm::SectionDecoder<uint64_t>>,
              "decode_model_id must be usable as a SectionDecoder<uint64_t>.");

hsm::Section inline_section(const WireId& wire) {
    hsm::ManifestEntry entry{};
    entry.tag = hsm::model_id_tag;
    std::copy(wire.begin(), wire.end(), entry.inline_bytes.begin());
    return hsm::Section{entry};
}

}  // namespace

TEST(HsmModelIdSectionTest, decode_reads_the_little_endian_wire_bytes) {
    EXPECT_EQ(hsm::decode_model_id(inline_section(sample_wire)), sample_id);
    EXPECT_EQ(hsm::decode_model_id(inline_section(max_wire)), max_id);
    EXPECT_EQ(hsm::decode_model_id(inline_section(WireId{})), 0u);
}

TEST(HsmModelIdSectionTest, decode_rejects_a_section_with_a_different_tag_id) {
    hsm::ManifestEntry entry{};
    entry.tag = hsm::SectionTag::make(hsm::model, /*is_inline=*/true);
    std::copy(sample_wire.begin(), sample_wire.end(), entry.inline_bytes.begin());

    EXPECT_FALSE(hsm::decode_model_id(hsm::Section{entry}).has_value());
}

TEST(HsmModelIdSectionTest, decode_rejects_a_pointer_mode_section) {
    hsm::ManifestEntry entry{};
    entry.tag = hsm::SectionTag::make(hsm::model_id, /*is_inline=*/false);
    entry.size = sample_wire.size();
    const hsm::Section section{entry, {reinterpret_cast<const std::byte*>(sample_wire.data()), sample_wire.size()}};

    EXPECT_FALSE(hsm::decode_model_id(section).has_value());
}

}  // namespace ov::test
