// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_weights.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/runtime/hsm_deferred_writer.hpp"

namespace ov::test {
namespace hsm = ov::runtime::hsm;

TEST(HsmWeightsTest, round_trips_through_stream) {
    const hsm::WeightsBuffer expected{
        std::byte{0x00},
        std::byte{0x01},
        std::byte{0x7F},
        std::byte{0x80},
        std::byte{0xFE},
        std::byte{0xFF},
    };
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    auto writer = hsm::DeferredWriter::open(stream);
    ASSERT_TRUE(writer.has_value());

    ASSERT_TRUE(writer->add_section(hsm::any_device_id,
                                    hsm::weights_tag,
                                    expected.size(),
                                    hsm::encode_weights({expected.data(), expected.size()})));

    ASSERT_FALSE(writer->finalize());
    stream.seekg(0);
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader.has_value());

    const auto decoded = reader->decode<hsm::WeightsBuffer>(hsm::Tags::weights, hsm::decode_weights);
    ASSERT_TRUE(decoded.has_value());
    EXPECT_EQ(*decoded, expected);
}

TEST(HsmWeightsTest, decodes_from_buffer_backed_reader) {
    const hsm::WeightsBuffer expected{
        std::byte{0x12},
        std::byte{0x34},
        std::byte{0x56},
    };
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    auto writer = hsm::DeferredWriter::open(stream);
    ASSERT_TRUE(writer.has_value());
    ASSERT_TRUE(writer->add_section(hsm::any_device_id,
                                    hsm::weights_tag,
                                    expected.size(),
                                    hsm::encode_weights({expected.data(), expected.size()})));
    ASSERT_FALSE(writer->finalize());

    const auto container = stream.str();
    const auto reader = hsm::Reader::open(reinterpret_cast<const uint8_t*>(container.data()), container.size());
    ASSERT_TRUE(reader.has_value());

    const auto decoded = reader->decode<hsm::WeightsBuffer>(hsm::Tags::weights, hsm::decode_weights);
    ASSERT_TRUE(decoded.has_value());
    EXPECT_EQ(*decoded, expected);

    const auto view = reader->decode<ov::util::MemoryView>(hsm::Tags::weights, hsm::decode_weights_view);
    ASSERT_TRUE(view.has_value());
    ASSERT_EQ(view->size(), expected.size());
    const auto section = reader->section(hsm::Tags::weights);
    ASSERT_TRUE(section.has_value());
    EXPECT_EQ(view->data(), section->view().data());
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), static_cast<const std::byte*>(view->data())));
}

TEST(HsmWeightsTest, preserves_multiple_weights_sections_in_order) {
    const std::vector<hsm::WeightsBuffer> expected{
        {std::byte{0x12}},
        {std::byte{0x34}, std::byte{0x56}},
        {std::byte{0x78}, std::byte{0x9A}, std::byte{0xBC}},
    };
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    auto writer = hsm::DeferredWriter::open(stream);
    ASSERT_TRUE(writer.has_value());
    for (const auto& weights : expected) {
        ASSERT_TRUE(writer->add_section(hsm::any_device_id,
                                        hsm::weights_tag,
                                        weights.size(),
                                        hsm::encode_weights({weights.data(), weights.size()})));
    }
    ASSERT_FALSE(writer->finalize());

    const auto container = stream.str();
    const auto reader = hsm::Reader::open(reinterpret_cast<const uint8_t*>(container.data()), container.size());
    ASSERT_TRUE(reader.has_value());
    ASSERT_EQ(reader->count(hsm::Tags::weights), expected.size());
    EXPECT_EQ(reader->decode<hsm::WeightsBuffer>(hsm::Tags::weights, hsm::decode_weights), expected.front());

    const auto first_view = reader->decode<ov::util::MemoryView>(hsm::Tags::weights, hsm::decode_weights_view);
    ASSERT_TRUE(first_view.has_value());
    EXPECT_TRUE(std::equal(expected.front().begin(), expected.front().end(), first_view->data()));

    const auto sections = reader->sections(hsm::Tags::weights);
    ASSERT_EQ(sections.size(), expected.size());
    for (size_t index = 0; index < sections.size(); ++index) {
        EXPECT_EQ(hsm::decode_weights(sections[index]), expected[index]);
    }
    EXPECT_EQ(reader->decode_all<hsm::WeightsBuffer>(hsm::Tags::weights, hsm::decode_weights), expected);
}

TEST(HsmWeightsTest, view_requires_addressable_storage) {
    const hsm::WeightsBuffer expected{std::byte{0x12}, std::byte{0x34}};
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    auto writer = hsm::DeferredWriter::open(stream);
    ASSERT_TRUE(writer.has_value());
    ASSERT_TRUE(writer->add_section(hsm::any_device_id,
                                    hsm::weights_tag,
                                    hsm::encode_weights({expected.data(), expected.size()})));
    ASSERT_FALSE(writer->finalize());

    stream.seekg(0);
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader.has_value());
    EXPECT_FALSE(reader->decode<ov::util::MemoryView>(hsm::Tags::weights, hsm::decode_weights_view).has_value());
    EXPECT_EQ(reader->decode<hsm::WeightsBuffer>(hsm::Tags::weights, hsm::decode_weights), expected);
}

TEST(HsmWeightsTest, decode_weights_returns_nullopt_when_stream_throws) {
    std::istringstream stream(std::string{}, std::ios::in | std::ios::binary);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    hsm::ManifestEntry entry{};
    entry.tag = hsm::weights_tag;
    entry.size = 1;
    const hsm::Section section(entry, stream, std::streampos{0});

    EXPECT_FALSE(hsm::decode_weights(section).has_value());
}

TEST(HsmWeightsTest, empty_weights_have_an_empty_view) {
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    auto writer = hsm::DeferredWriter::open(stream);
    ASSERT_TRUE(writer.has_value());
    ASSERT_TRUE(writer->add_section(hsm::any_device_id, hsm::weights_tag, 0, hsm::encode_weights({})));
    ASSERT_FALSE(writer->finalize());

    const auto container = stream.str();
    const auto reader = hsm::Reader::open(reinterpret_cast<const uint8_t*>(container.data()), container.size());
    ASSERT_TRUE(reader.has_value());
    const auto view = reader->decode<ov::util::MemoryView>(hsm::Tags::weights, hsm::decode_weights_view);
    ASSERT_TRUE(view.has_value());
    EXPECT_TRUE(view->empty());
    EXPECT_EQ(reader->decode<hsm::WeightsBuffer>(hsm::Tags::weights, hsm::decode_weights), hsm::WeightsBuffer{});
}

TEST(HsmWeightsTest, view_rejects_inline_entry_with_weights_tag_id) {
    const std::byte value{0x42};
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    auto writer = hsm::DeferredWriter::open(stream);
    ASSERT_TRUE(writer.has_value());
    const auto inline_weights = hsm::SectionTag::make(hsm::weights, /*is_inline=*/true);
    ASSERT_TRUE(writer->add_section(hsm::any_device_id, inline_weights, {&value, 1}));
    ASSERT_FALSE(writer->finalize());

    const auto container = stream.str();
    const auto reader = hsm::Reader::open(reinterpret_cast<const uint8_t*>(container.data()), container.size());
    ASSERT_TRUE(reader.has_value());
    EXPECT_FALSE(reader->decode<ov::util::MemoryView>(hsm::Tags::weights, hsm::decode_weights_view).has_value());
}

TEST(HsmWeightsTest, preserves_bytes_with_requested_section_alignment) {
    const hsm::WeightsBuffer expected{
        std::byte{0x00},
        std::byte{0x7F},
        std::byte{0x80},
        std::byte{0xFF},
    };
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    auto writer = hsm::DeferredWriter::open(stream);
    ASSERT_TRUE(writer.has_value());
    ASSERT_TRUE(writer->add_section(hsm::any_device_id,
                                    hsm::weights_tag,
                                    expected.size(),
                                    hsm::encode_weights({expected.data(), expected.size()}),
                                    {/*offset_align=*/4096, /*size_align=*/4096}));
    ASSERT_FALSE(writer->finalize());

    stream.seekg(0);
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader.has_value());
    ASSERT_EQ(reader->count(hsm::weights), 1u);

    const auto entry = std::find_if(reader->entries().begin(), reader->entries().end(), [](const auto& candidate) {
        return candidate.device == hsm::any_device_id && candidate.tag.id() == hsm::weights;
    });
    ASSERT_NE(entry, reader->entries().end());
    EXPECT_EQ(entry->offset % 4096, 0u);

    const auto decoded = reader->decode<hsm::WeightsBuffer>(hsm::Tags::weights, hsm::decode_weights);
    ASSERT_TRUE(decoded.has_value());
    EXPECT_EQ(*decoded, expected);
}

TEST(HsmWeightsTest, returns_nullopt_when_weights_section_is_absent) {
    const std::byte model_id{0x42};
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    auto writer = hsm::DeferredWriter::open(stream);
    ASSERT_TRUE(writer.has_value());
    ASSERT_TRUE(writer->add_section(hsm::any_device_id, hsm::model_id_tag, {&model_id, 1}));
    ASSERT_FALSE(writer->finalize());

    stream.seekg(0);
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader.has_value());
    EXPECT_FALSE(reader->decode<hsm::WeightsBuffer>(hsm::Tags::weights, hsm::decode_weights).has_value());
    EXPECT_FALSE(reader->decode<ov::util::MemoryView>(hsm::Tags::weights, hsm::decode_weights_view).has_value());
}

}  // namespace ov::test
