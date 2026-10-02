// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_deferred_writer.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "openvino/core/except.hpp"

// DeferredWriter-specific tests
namespace ov::test {
namespace hsm = ov::runtime::hsm;
namespace {

constexpr size_t k_buffer_capacity = 4096;

ov::util::MemoryView view_of(const std::string& s) {
    return {reinterpret_cast<const std::byte*>(s.data()), s.size()};
}

hsm::DeferredWriter open_writer(std::ostream& stream) {
    auto writer = hsm::DeferredWriter::open(stream);
    EXPECT_TRUE(writer.has_value());
    return std::move(writer).value();
}

hsm::DeferredWriter open_buffer_writer(std::byte* dst, size_t size) {
    auto writer = hsm::DeferredWriter::open(dst, size);
    EXPECT_TRUE(writer.has_value());
    return std::move(writer).value();
}

// Raw container parsing, no Reader dependency - same pattern as hsm_writer.cpp, kept file-local on purpose.
struct ParsedContainer {
    hsm::Header header{};
    std::vector<hsm::ManifestEntry> entries;
    std::vector<std::byte> bytes;
};

std::optional<ParsedContainer> parse_container(std::vector<std::byte> raw) {
    if (raw.size() < sizeof(hsm::Header)) {
        return std::nullopt;
    }
    hsm::Header header{};
    std::memcpy(&header, raw.data(), sizeof(header));
    if (!hsm::is_valid_header_fields(header)) {
        return std::nullopt;
    }
    std::vector<hsm::ManifestEntry> entries(header.manifest_size / sizeof(hsm::ManifestEntry));
    std::memcpy(entries.data(), raw.data() + header.manifest_offset, header.manifest_size);
    return ParsedContainer{header, std::move(entries), std::move(raw)};
}

std::optional<ParsedContainer> parse_container(const std::string& stream_bytes) {
    const auto* data = reinterpret_cast<const std::byte*>(stream_bytes.data());
    return parse_container(std::vector<std::byte>(data, data + stream_bytes.size()));
}

std::optional<ParsedContainer> parse_container(const std::byte* data, size_t size) {
    return parse_container(std::vector<std::byte>(data, data + size));
}

const hsm::ManifestEntry* find_entry(const ParsedContainer& container, hsm::DeviceId device, uint32_t tag_id) {
    for (const auto& entry : container.entries) {
        if (entry.device == device && entry.tag.id() == tag_id) {
            return &entry;
        }
    }
    return nullptr;
}

std::string payload_string(const ParsedContainer& container, const hsm::ManifestEntry& entry) {
    if (entry.tag.is_inline()) {
        return {reinterpret_cast<const char*>(entry.inline_bytes.data()), entry.inline_bytes.size()};
    }
    return {reinterpret_cast<const char*>(container.bytes.data() + entry.offset), entry.size};
}

}  // namespace

static_assert(std::is_move_constructible_v<hsm::DeferredWriter>);
static_assert(std::is_move_assignable_v<hsm::DeferredWriter>);
static_assert(!std::is_copy_constructible_v<hsm::DeferredWriter>);
static_assert(!std::is_copy_assignable_v<hsm::DeferredWriter>);

// --- open() factories ---

TEST(HsmDeferredWriterTest, open_stream_rejects_an_already_failed_stream) {
    std::stringstream stream;
    stream.setstate(std::ios::badbit);
    EXPECT_FALSE(hsm::DeferredWriter::open(stream).has_value());
}

TEST(HsmDeferredWriterTest, open_buffer_rejects_a_buffer_too_small_for_the_header) {
    std::vector<std::byte> too_small(sizeof(hsm::Header) - 1);
    EXPECT_FALSE(hsm::DeferredWriter::open(too_small.data(), too_small.size()).has_value());
}

TEST(HsmDeferredWriterTest, open_buffer_rejects_a_null_buffer) {
    EXPECT_FALSE(hsm::DeferredWriter::open(nullptr, 0).has_value());
}

// --- Writing into the two destination kinds ---

TEST(HsmDeferredWriterTest, writes_into_a_preallocated_buffer) {
    const std::string model = "model-bytes";

    std::vector<std::byte> buffer(k_buffer_capacity);
    auto writer = open_buffer_writer(buffer.data(), buffer.size());
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(model));
    ASSERT_FALSE(writer.finalize());

    const auto container = parse_container(buffer.data(), buffer.size());
    ASSERT_TRUE(container.has_value());
    const auto* entry = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(payload_string(*container, *entry), "model-bytes");
}

TEST(HsmDeferredWriterTest, finalize_reports_write_failed_for_an_undersized_buffer) {
    std::vector<std::byte> too_small(sizeof(hsm::Header));  // no room for even one section
    auto writer = open_buffer_writer(too_small.data(), too_small.size());
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(std::string("model")));
    EXPECT_EQ(writer.finalize(), hsm::WriteErrc::write_failed);
}

TEST(HsmDeferredWriterTest, finalize_reports_write_failed_when_the_stream_is_already_bad) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(std::string("model")));
    stream.setstate(std::ios::badbit);  // fails every subsequent write
    EXPECT_EQ(writer.finalize(), hsm::WriteErrc::write_failed);
}

// --- add_section() validates its inputs ---

TEST(HsmDeferredWriterTest, sized_section_encoder_rejects_an_inline_mode_tag) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    EXPECT_THROW(writer.add_section(hsm::any_device_id, hsm::model_id_tag, 4, [](const hsm::SectionSink&) {}),
                 ov::AssertFailure);
}

TEST(HsmDeferredWriterTest, unsized_section_encoder_rejects_an_inline_mode_tag) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    EXPECT_THROW(writer.add_section(hsm::any_device_id, hsm::model_id_tag, [](const hsm::SectionSink&) {}),
                 ov::AssertFailure);
}

// --- finalize() semantics ---

TEST(HsmDeferredWriterTest, finalize_is_idempotent) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(std::string("model")));
    const auto first = writer.finalize();
    const auto written_once = stream.str();
    const auto second = writer.finalize();

    EXPECT_EQ(first, second);
    EXPECT_EQ(stream.str(), written_once);  // no bytes written again
}

TEST(HsmDeferredWriterTest, unsized_section_forces_the_header_to_be_patched_after_the_fact) {
    // With only sized content, the header is computed in one pass up front. An unsized section forces the
    // placeholder-then-patch fallback - verified here by confirming the final header is still correct.
    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id, hsm::model_tag, [](const hsm::SectionSink& sink) {
        const std::string content = "discovered-at-write-time";
        sink({reinterpret_cast<const std::byte*>(content.data()), content.size()});
    });
    ASSERT_FALSE(writer.finalize());

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    EXPECT_TRUE(hsm::is_valid_header_fields(container->header));
    const auto* entry = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(payload_string(*container, *entry), "discovered-at-write-time");
}

}  // namespace ov::test
