// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_model_struct.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace ov::test {
namespace hsm = ov::runtime::hsm;
namespace {

std::string collect(const hsm::ISectionEncoder& encoder) {
    std::string encoded;
    encoder.encode([&encoded](ov::util::MemoryView chunk) {
        encoded.append(reinterpret_cast<const char*>(chunk.data()), chunk.size());
    });
    return encoded;
}

std::shared_ptr<ov::AlignedBuffer> buffer_of(const std::string& bytes) {
    if (bytes.empty()) {
        return std::make_shared<ov::AlignedBuffer>();  // AlignedBuffer(0) would hold one byte
    }
    auto buffer = std::make_shared<ov::AlignedBuffer>(bytes.size());
    std::memcpy(buffer->get_ptr(), bytes.data(), bytes.size());
    return buffer;
}

hsm::ModelStructEncoder::Producer write(std::string bytes) {
    return [bytes = std::move(bytes)] {
        return buffer_of(bytes);
    };
}

std::string text_of(const ov::AlignedBuffer& buffer) {
    return buffer.size() ? std::string{buffer.get_ptr<char>(), buffer.size()} : std::string{};
}

}  // namespace

static_assert(
    std::is_convertible_v<decltype(&hsm::decode_model_struct), hsm::SectionDecoder<std::shared_ptr<ov::AlignedBuffer>>>,
    "decode_model_struct must be usable as a SectionDecoder<std::shared_ptr<ov::AlignedBuffer>>.");

TEST(HsmModelStructTest, round_trips_xml) {
    const std::string expected = "<?xml version=\"1.0\"?><net name=\"test\">\n<layers/>\n</net>";
    const auto encoded = collect(hsm::ModelStructEncoder{write(expected)});
    EXPECT_EQ(encoded, expected);

    hsm::ManifestEntry entry{};
    entry.tag = hsm::model_tag;
    entry.size = encoded.size();
    const hsm::Section section{entry, {reinterpret_cast<const std::byte*>(encoded.data()), encoded.size()}};
    const auto decoded = hsm::decode_model_struct(section);
    ASSERT_TRUE(decoded);
    EXPECT_EQ(text_of(**decoded), expected);
}

TEST(HsmModelStructTest, round_trips_empty_xml) {
    EXPECT_TRUE(collect(hsm::ModelStructEncoder{write({})}).empty());

    hsm::ManifestEntry entry{};
    entry.tag = hsm::model_tag;
    const hsm::Section section{entry, ov::util::MemoryView{}};
    const auto decoded = hsm::decode_model_struct(section);
    ASSERT_TRUE(decoded);
    EXPECT_EQ((*decoded)->size(), 0u);
}

TEST(HsmModelStructTest, encoder_generates_structure_at_encode_time) {
    auto xml = std::make_shared<std::string>();
    const hsm::ModelStructEncoder encoder{[xml] {
        return buffer_of(*xml);
    }};
    *xml = "<net name=\"deferred\"/>";

    EXPECT_EQ(collect(encoder), "<net name=\"deferred\"/>");
}

TEST(HsmModelStructTest, encoder_writes_same_bytes_on_every_call) {
    const std::string expected = "<net name=\"retry\"/>";
    const hsm::ModelStructEncoder encoder{write(expected)};

    EXPECT_EQ(collect(encoder), expected);
    EXPECT_EQ(collect(encoder), expected);
}

TEST(HsmModelStructTest, encoder_rejects_empty_producer) {
    EXPECT_THROW(hsm::ModelStructEncoder{nullptr}, ov::AssertFailure);
}

TEST(HsmModelStructTest, encoder_rejects_null_buffer) {
    const hsm::ModelStructEncoder encoder{[] {
        return std::shared_ptr<ov::AlignedBuffer>{};
    }};
    EXPECT_THROW(collect(encoder), ov::AssertFailure);
}

TEST(HsmModelStructTest, decodes_stream_section) {
    const std::string expected = "<net/>";
    std::istringstream stream{expected};
    hsm::ManifestEntry entry{};
    entry.tag = hsm::model_tag;
    entry.size = expected.size();
    const hsm::Section section{entry, stream, std::streampos{0}};

    const auto decoded = hsm::decode_model_struct(section);
    ASSERT_TRUE(decoded);
    EXPECT_EQ(text_of(**decoded), expected);
}

TEST(HsmModelStructTest, decode_is_zero_copy_for_addressable_section) {
    const std::string expected = "<net name=\"zero-copy\"/>";
    const auto blob = buffer_of(expected);
    hsm::ManifestEntry entry{};
    entry.tag = hsm::model_tag;
    entry.size = expected.size();
    const hsm::Section section{entry, {blob->get_ptr<std::byte>(), blob->size()}};

    const auto decoded = hsm::decode_model_struct(section);
    ASSERT_TRUE(decoded);
    EXPECT_EQ((*decoded)->get_ptr(), blob->get_ptr());
    EXPECT_EQ(text_of(**decoded), expected);
}

TEST(HsmModelStructTest, decode_reports_truncated_stream_section) {
    std::istringstream stream{"abc"};
    hsm::ManifestEntry entry{};
    entry.tag = hsm::model_tag;
    entry.size = 8;
    const hsm::Section section{entry, stream, std::streampos{0}};

    EXPECT_FALSE(hsm::decode_model_struct(section).has_value());
}

}  // namespace ov::test
