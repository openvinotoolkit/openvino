// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_sections.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>

#include "openvino/runtime/hsm_deferred_writer.hpp"
#include "openvino/runtime/hsm_format.hpp"
#include "openvino/runtime/hsm_reader.hpp"

// Core section codecs, each round-tripped through a real Writer and a real Reader rather than a
// hand-built container - so the encode/decode pair is exercised against the bytes a writer really emits.
namespace ov::test {
namespace hsm = ov::runtime::hsm;
namespace {

// `id` is borrowed by the writer until finalize(), which happens here - so a by-value copy is enough.
std::string container_with_model_id(hsm::DeviceId device, uint64_t id) {
    std::stringstream stream;
    auto writer = hsm::DeferredWriter::open(stream);
    EXPECT_TRUE(writer.has_value());
    EXPECT_TRUE(writer->add_section(device, hsm::model_id_tag, hsm::encode_model_id(id)));
    EXPECT_EQ(writer->finalize(), std::error_code{});
    return stream.str();
}

std::optional<hsm::Reader> open_reader(const std::string& bytes) {
    return hsm::Reader::open(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
}

}  // namespace

TEST(HsmModelIdSectionTest, round_trips_through_a_real_writer_and_reader) {
    constexpr uint64_t id = 0x0123456789ABCDEF;

    const auto bytes = container_with_model_id(hsm::any_device_id, id);
    const auto reader = open_reader(bytes);
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section(hsm::Tags::model_id);
    ASSERT_TRUE(section.has_value());
    // The slot reads back full-width whatever the payload was, which is why decode reads a fixed 8 bytes.
    EXPECT_GT(section->size(), sizeof(uint64_t));

    const auto payload = section->view();
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(hsm::decode_model_id(*payload), id);
}

TEST(HsmModelIdSectionTest, inline_section_adds_no_payload_bytes_to_the_container) {
    const auto bytes = container_with_model_id(hsm::any_device_id, 0x0123456789ABCDEF);

    // model_id is inline-mode, so its payload lives in the manifest entry itself - the manifest must start
    // right after the header, with no section-payload bytes written in between.
    const auto& header = hsm::Header::view(reinterpret_cast<const uint8_t*>(bytes.data()));
    EXPECT_EQ(header.manifest_offset, sizeof(hsm::Header));
}

TEST(HsmModelIdSectionTest, round_trips_the_maximum_id) {
    constexpr uint64_t id = std::numeric_limits<uint64_t>::max();

    const auto bytes = container_with_model_id(hsm::any_device_id, id);
    const auto reader = open_reader(bytes);
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section(hsm::Tags::model_id);
    ASSERT_TRUE(section.has_value());
    const auto payload = section->view();
    ASSERT_TRUE(payload.has_value());
    EXPECT_EQ(hsm::decode_model_id(*payload), id);
}

TEST(HsmModelIdSectionTest, decode_rejects_a_payload_too_short_to_hold_an_id) {
    const std::array<std::byte, sizeof(uint64_t) - 1> too_short{};

    EXPECT_FALSE(hsm::decode_model_id({too_short.data(), too_short.size()}).has_value());
    EXPECT_FALSE(hsm::decode_model_id({}).has_value());
}

TEST(HsmModelIdSectionTest, device_scoped_id_decodes_independently_of_the_common_one) {
    constexpr uint64_t common_id = 0x1111111111111111;
    constexpr uint64_t npu_id = 0x2222222222222222;

    std::stringstream stream;
    auto writer = hsm::DeferredWriter::open(stream);
    ASSERT_TRUE(writer.has_value());
    ASSERT_TRUE(writer->add_section(hsm::any_device_id, hsm::model_id_tag, hsm::encode_model_id(common_id)));
    ASSERT_TRUE(writer->add_section(hsm::npu_device_id, hsm::model_id_tag, hsm::encode_model_id(npu_id)));
    ASSERT_EQ(writer->finalize(), std::error_code{});

    const auto bytes = stream.str();
    const auto reader = open_reader(bytes);
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
