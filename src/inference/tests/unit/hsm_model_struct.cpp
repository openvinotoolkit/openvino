// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_model_struct.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace ov::test {
namespace hsm = ov::runtime::hsm;
namespace {

std::optional<std::string> round_trip(std::string xml) {
    const auto encoder = hsm::encode_model_struct(std::move(xml));
    std::string encoded;
    encoder->encode([&encoded](ov::util::MemoryView chunk) {
        encoded.append(reinterpret_cast<const char*>(chunk.data()), chunk.size());
    });

    hsm::ManifestEntry entry{};
    entry.tag = hsm::model_struct_tag;
    entry.size = encoded.size();
    const hsm::Section section{
        entry,
        ov::util::MemoryView{reinterpret_cast<const std::byte*>(encoded.data()), encoded.size()}};
    return hsm::decode_model_struct(section);
}

}  // namespace

TEST(HsmModelStructTest, round_trips_xml) {
    const std::string expected = "<?xml version=\"1.0\"?><net name=\"test\">\n<layers/>\n</net>";
    EXPECT_EQ(round_trip(expected), expected);
}

TEST(HsmModelStructTest, round_trips_empty_xml) {
    EXPECT_EQ(round_trip({}), std::string{});
}

TEST(HsmModelStructTest, views_xml_without_copying) {
    const std::string expected = "<net name=\"test\"/>";
    hsm::ManifestEntry entry{};
    entry.tag = hsm::model_struct_tag;
    entry.size = expected.size();
    const hsm::Section section{
        entry,
        ov::util::MemoryView{reinterpret_cast<const std::byte*>(expected.data()), expected.size()}};

    const auto decoded = hsm::decode_model_struct_view(section);
    ASSERT_TRUE(decoded.has_value());
    EXPECT_EQ(*decoded, expected);
    EXPECT_EQ(decoded->data(), expected.data());
}

TEST(HsmModelStructTest, views_empty_xml) {
    hsm::ManifestEntry entry{};
    entry.tag = hsm::model_struct_tag;
    const hsm::Section section{entry, ov::util::MemoryView{}};

    EXPECT_EQ(hsm::decode_model_struct_view(section), std::string_view{});
}

TEST(HsmModelStructTest, view_requires_addressable_pointer_section) {
    hsm::ManifestEntry entry{};
    entry.tag = hsm::model_struct_tag;
    entry.size = 3;
    std::istringstream stream{"xml"};
    const hsm::Section stream_section{entry, stream, std::streampos{0}};
    EXPECT_FALSE(hsm::decode_model_struct_view(stream_section).has_value());

    entry.tag = hsm::SectionTag::make(hsm::model_struct, /*is_inline=*/true);
    const hsm::Section inline_section{entry};
    EXPECT_FALSE(hsm::decode_model_struct_view(inline_section).has_value());
}

}  // namespace ov::test
