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
#include <initializer_list>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

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

struct InlineModelId {
    hsm::DeviceId device;
    WireId wire;
};

// Header followed directly by one inline model_id manifest entry per element - no section payloads.
std::vector<uint8_t> container_with_model_ids(std::initializer_list<InlineModelId> ids) {
    hsm::Header header{};
    header.magic = hsm::BlobMagic::single;
    header.version_major = hsm::FormatVersion::major;
    header.version_minor = hsm::FormatVersion::minor;
    header.manifest_offset = sizeof(hsm::Header);
    header.manifest_size = ids.size() * sizeof(hsm::ManifestEntry);
    header.container_size = header.manifest_offset + header.manifest_size;

    std::vector<uint8_t> blob(header.container_size);
    std::memcpy(blob.data(), &header, sizeof(header));
    auto* entry_bytes = blob.data() + header.manifest_offset;
    for (const auto& id : ids) {
        hsm::ManifestEntry entry{};
        entry.device = id.device;
        entry.tag = hsm::model_id_tag;
        std::copy(id.wire.begin(), id.wire.end(), entry.inline_bytes.begin());
        std::memcpy(entry_bytes, &entry, sizeof(entry));
        entry_bytes += sizeof(entry);
    }
    return blob;
}

void expect_encodes_to(const uint64_t& id, const WireId& expected) {
    const auto encoded = hsm::encode_model_id(id);
    ASSERT_EQ(encoded.size(), expected.size());
    EXPECT_EQ(std::memcmp(encoded.data(), expected.data(), expected.size()), 0);
}

}  // namespace

TEST(HsmModelIdSectionTest, encode_emits_the_little_endian_wire_bytes) {
    expect_encodes_to(sample_id, sample_wire);
    expect_encodes_to(max_id, max_wire);
    expect_encodes_to(0, WireId{});
}

TEST(HsmModelIdSectionTest, decodes_an_id_from_a_hand_built_container) {
    const auto bytes = container_with_model_ids({{hsm::any_device_id, sample_wire}});
    const auto reader = hsm::Reader::open(bytes.data(), bytes.size());
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section(hsm::Tags::model_id);
    ASSERT_TRUE(section.has_value());
    // The slot reads back full-width whatever the payload was, which is why decode reads a fixed 8 bytes.
    EXPECT_GT(section->size(), sizeof(uint64_t));

    const auto payload = section->view();
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(hsm::decode_model_id(*payload), sample_id);
}

TEST(HsmModelIdSectionTest, decodes_the_maximum_id_from_a_hand_built_container_stream) {
    const auto bytes = container_with_model_ids({{hsm::any_device_id, max_wire}});
    std::stringstream stream{std::string{bytes.begin(), bytes.end()}};
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section(hsm::Tags::model_id);
    ASSERT_TRUE(section.has_value());
    const auto payload = section->view();
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(hsm::decode_model_id(*payload), max_id);
}

TEST(HsmModelIdSectionTest, decode_rejects_a_payload_too_short_to_hold_an_id) {
    const std::array<std::byte, sizeof(uint64_t) - 1> too_short{};

    EXPECT_FALSE(hsm::decode_model_id({too_short.data(), too_short.size()}).has_value());
    EXPECT_FALSE(hsm::decode_model_id({}).has_value());
}

TEST(HsmModelIdSectionTest, device_scoped_id_decodes_independently_of_the_common_one) {
    constexpr uint64_t common_id = 0x1122334455667788;
    constexpr WireId common_wire{0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11};
    constexpr uint64_t npu_id = 0xA1B2C3D4E5F60718;
    constexpr WireId npu_wire{0x18, 0x07, 0xF6, 0xE5, 0xD4, 0xC3, 0xB2, 0xA1};

    const auto bytes = container_with_model_ids({{hsm::any_device_id, common_wire}, {hsm::npu_device_id, npu_wire}});
    const auto reader = hsm::Reader::open(bytes.data(), bytes.size());
    ASSERT_TRUE(reader.has_value());

    const auto common = reader->section(hsm::Tags::model_id);
    ASSERT_TRUE(common.has_value());
    EXPECT_EQ(hsm::decode_model_id(*common->view()), common_id);

    const auto overridden = reader->section(hsm::npu_device_id, hsm::Tags::model_id);
    ASSERT_TRUE(overridden.has_value());
    EXPECT_EQ(hsm::decode_model_id(*overridden->view()), npu_id);

    // A device that published no id of its own falls back to the common one.
    EXPECT_FALSE(reader->section(hsm::gpu_device_id, hsm::Tags::model_id).has_value());
}

}  // namespace ov::test
