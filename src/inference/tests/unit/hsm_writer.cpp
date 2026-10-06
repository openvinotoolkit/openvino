// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_writer.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"

namespace ov::test {
namespace hsm = ov::runtime::hsm;
namespace {

constexpr hsm::DeviceId fake_device_id = 7;
constexpr size_t k_buffer_capacity = 4096;  // generous fixed-buffer size for tests needing raw byte access

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

// Raw container parsing, no Reader dependency: just enough (Header + manifest entries) to check what the
// writer actually produced; payload bytes are sliced directly out of the raw buffer.
struct ParsedContainer {
    hsm::Header header{};
    std::vector<hsm::ManifestEntry> entries;
    std::vector<std::byte> bytes;  // whole container - pointer-mode payloads are sliced from this
};

std::optional<ParsedContainer> parse_container(std::vector<std::byte> raw) {
    if (raw.size() < sizeof(hsm::Header)) {
        return std::nullopt;
    }
    const auto& header = hsm::Header::view(reinterpret_cast<const uint8_t*>(raw.data()));
    if (!hsm::is_valid_header_fields(header)) {
        return std::nullopt;
    }
    const auto first_entry = reinterpret_cast<const hsm::ManifestEntry*>(raw.data() + header.manifest_offset);
    std::vector<hsm::ManifestEntry> entries(first_entry,
                                            first_entry + header.manifest_size / sizeof(hsm::ManifestEntry));
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

std::vector<const hsm::ManifestEntry*> find_entries(const ParsedContainer& container,
                                                    hsm::DeviceId device,
                                                    uint32_t tag_id) {
    std::vector<const hsm::ManifestEntry*> found;
    for (const auto& entry : container.entries) {
        if (entry.device == device && entry.tag.id() == tag_id) {
            found.push_back(&entry);
        }
    }
    return found;
}

std::string payload_string(const ParsedContainer& container, const hsm::ManifestEntry& entry) {
    if (entry.tag.is_inline()) {
        return {reinterpret_cast<const char*>(entry.inline_bytes.data()), entry.inline_bytes.size()};
    }
    return {reinterpret_cast<const char*>(container.bytes.data() + entry.offset), entry.size};
}

}  // namespace

// --- add_section() view overload ---

TEST(HsmWriterTest, empty_container_is_valid_and_has_no_sections) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    EXPECT_FALSE(writer.finalize());

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    EXPECT_TRUE(container->entries.empty());
    EXPECT_EQ(find_entry(*container, hsm::any_device_id, hsm::model), nullptr);
}

TEST(HsmWriterTest, round_trips_inline_and_pointer_sections) {
    const std::string id = "id";
    const std::string model = "compiled-model-bytes";

    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id, hsm::model_id_tag, view_of(id));  // inline-mode Core tag
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(model));  // pointer-mode Core tag
    ASSERT_FALSE(writer.finalize());

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());

    // Inline entries always carry the full 24-byte slot; the writer zero-fills past the payload.
    const auto* read_id = find_entry(*container, hsm::any_device_id, hsm::model_id);
    ASSERT_NE(read_id, nullptr);
    EXPECT_TRUE(read_id->tag.is_inline());
    EXPECT_EQ(payload_string(*container, *read_id).substr(0, 2), "id");

    const auto* read_model = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(read_model, nullptr);
    EXPECT_TRUE(read_model->tag.is_pointer());
    EXPECT_EQ(payload_string(*container, *read_model), "compiled-model-bytes");
}

TEST(HsmWriterTest, tag_reserved_bytes_round_trip_and_default_to_zero) {
    const std::string payload = "device-payload";
    const auto shard_tag = hsm::SectionTag::make_device_tag(/*local_id=*/1, /*is_inline=*/false);
    const std::array<uint8_t, 4> reserved{0x01, 0x02, 0x03, 0x04};

    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(fake_device_id, hsm::SectionTagReserved{shard_tag, reserved}, view_of(payload));
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(payload));  // bare SectionTag still compiles
    ASSERT_FALSE(writer.finalize());

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());

    const auto* shard = find_entry(*container, fake_device_id, shard_tag.id());
    ASSERT_NE(shard, nullptr);
    EXPECT_EQ(shard->tag_reserved, reserved);

    const auto* model = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->tag_reserved, (std::array<uint8_t, 4>{}));
}

TEST(HsmWriterTest, device_specific_section_is_scoped_to_its_device) {
    const std::string payload = "device-payload";
    const auto shard_tag = hsm::SectionTag::make_device_tag(/*local_id=*/1, /*is_inline=*/false);

    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(fake_device_id, shard_tag, view_of(payload));
    ASSERT_FALSE(writer.finalize());

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());

    EXPECT_EQ(find_entry(*container, hsm::any_device_id, shard_tag.id()), nullptr);  // scoped to its device
    const auto* shard = find_entry(*container, fake_device_id, shard_tag.id());
    ASSERT_NE(shard, nullptr);
    EXPECT_EQ(payload_string(*container, *shard), "device-payload");
}

TEST(HsmWriterTest, aligns_pointer_section_offset_and_pads_slot_to_aligned_size) {
    const std::string weights = "weights";  // 7 bytes; slot padded to a multiple of 64
    const std::string tail = "tail";
    const auto tail_tag = hsm::SectionTag::make_device_tag(9, false);

    std::vector<std::byte> buffer(k_buffer_capacity);
    auto writer = open_buffer_writer(buffer.data(), buffer.size());
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(weights), {/*offset_align=*/64});
    // offset_align 1: its offset reveals whether the previous slot's size was padded to 64.
    writer.add_section(fake_device_id, tail_tag, view_of(tail));
    ASSERT_FALSE(writer.finalize());

    const auto container = parse_container(buffer.data(), buffer.size());
    ASSERT_TRUE(container.has_value());

    const auto* weights_entry = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(weights_entry, nullptr);
    EXPECT_EQ(weights_entry->offset % 64, 0u);
    EXPECT_EQ(payload_string(*container, *weights_entry), "weights");

    const auto* tail_entry = find_entry(*container, fake_device_id, tail_tag.id());
    ASSERT_NE(tail_entry, nullptr);
    // Starts after the previous slot padded up to 64 bytes, not right after the 7 payload bytes.
    EXPECT_EQ(tail_entry->offset, weights_entry->offset + 64);
}

TEST(HsmWriterTest, zero_alignment_behaves_the_same_as_one) {
    const std::string a = "aa";
    const std::string b = "bbb";
    const auto b_tag = hsm::SectionTag::make_device_tag(4, false);

    std::stringstream zero_stream;
    auto zero_writer = open_writer(zero_stream);
    zero_writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(a), {/*offset_align=*/0});
    zero_writer.add_section(fake_device_id, b_tag, view_of(b), {/*offset_align=*/0});
    ASSERT_FALSE(zero_writer.finalize());

    std::stringstream one_stream;
    auto one_writer = open_writer(one_stream);
    one_writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(a), {/*offset_align=*/1});
    one_writer.add_section(fake_device_id, b_tag, view_of(b), {/*offset_align=*/1});
    ASSERT_FALSE(one_writer.finalize());

    EXPECT_EQ(zero_stream.str(), one_stream.str());
}

TEST(HsmWriterTest, size_align_pads_the_slot_independently_of_offset_align) {
    const std::string weights = "weights";  // 7 bytes
    const std::string tail = "tail";
    const auto tail_tag = hsm::SectionTag::make_device_tag(9, false);

    std::vector<std::byte> buffer(k_buffer_capacity);
    auto writer = open_buffer_writer(buffer.data(), buffer.size());
    // Offset aligned to 64, but the slot itself only needs to be padded to a multiple of 16.
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(weights), {/*offset_align=*/64, /*size_align=*/16});
    writer.add_section(fake_device_id, tail_tag, view_of(tail));
    ASSERT_FALSE(writer.finalize());

    const auto container = parse_container(buffer.data(), buffer.size());
    ASSERT_TRUE(container.has_value());

    const auto* weights_entry = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(weights_entry, nullptr);
    EXPECT_EQ(weights_entry->offset % 64, 0u);

    const auto* tail_entry = find_entry(*container, fake_device_id, tail_tag.id());
    ASSERT_NE(tail_entry, nullptr);
    // Slot padded to 16 (not 64, which the default single-value behavior would have produced).
    EXPECT_EQ(tail_entry->offset, weights_entry->offset + 16);
}

TEST(HsmWriterTest, pads_an_alignment_gap_larger_than_the_internal_zero_fill_chunk) {
    // write_zeros() fills padding gaps via a small internal buffer reused in a loop - this forces a gap
    // bigger than that buffer, exercising more than one iteration of it.
    const std::string a = "a";
    const std::string b = "b";
    const auto b_tag = hsm::SectionTag::make_device_tag(5, false);

    std::vector<std::byte> buffer(k_buffer_capacity);
    auto writer = open_buffer_writer(buffer.data(), buffer.size());
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(a), {/*offset_align=*/1024});
    writer.add_section(fake_device_id, b_tag, view_of(b), {/*offset_align=*/1024});
    ASSERT_FALSE(writer.finalize());

    const auto container = parse_container(buffer.data(), buffer.size());
    ASSERT_TRUE(container.has_value());
    const auto* a_entry = find_entry(*container, hsm::any_device_id, hsm::model);
    const auto* b_entry = find_entry(*container, fake_device_id, b_tag.id());
    ASSERT_NE(a_entry, nullptr);
    ASSERT_NE(b_entry, nullptr);
    EXPECT_EQ(b_entry->offset, a_entry->offset + 1024);
    EXPECT_EQ(payload_string(*container, *a_entry), "a");
    EXPECT_EQ(payload_string(*container, *b_entry), "b");
}

TEST(HsmWriterTest, preserves_multiple_sections_sharing_one_tag_in_order) {
    const std::string s0 = "shard-0";
    const std::string s1 = "shard-1";
    const auto shard_tag = hsm::SectionTag::make_device_tag(/*local_id=*/2, /*is_inline=*/false);

    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(fake_device_id, shard_tag, view_of(s0));
    writer.add_section(fake_device_id, shard_tag, view_of(s1));
    ASSERT_FALSE(writer.finalize());

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    const auto shards = find_entries(*container, fake_device_id, shard_tag.id());
    ASSERT_EQ(shards.size(), 2u);
    EXPECT_EQ(payload_string(*container, *shards[0]), "shard-0");
    EXPECT_EQ(payload_string(*container, *shards[1]), "shard-1");
}

TEST(HsmWriterTest, rejects_inline_payload_exceeding_entry_capacity) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    const std::string too_big(25, 'x');  // inline capacity is 24 bytes
    EXPECT_THROW(writer.add_section(hsm::any_device_id, hsm::model_id_tag, view_of(too_big)), ov::AssertFailure);
}

TEST(HsmWriterTest, rejects_non_power_of_two_alignment) {
    const std::string x = "x";
    std::stringstream stream;
    auto writer = open_writer(stream);
    EXPECT_THROW(writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(x), {/*offset_align=*/3}),
                 ov::AssertFailure);
}

TEST(HsmWriterTest, inline_section_ignores_a_non_power_of_two_alignment) {
    const std::string id = "x";
    std::stringstream stream;
    auto writer = open_writer(stream);
    EXPECT_NO_THROW(writer.add_section(hsm::any_device_id, hsm::model_id_tag, view_of(id), {/*offset_align=*/3}));
    EXPECT_FALSE(writer.finalize());
}

TEST(HsmWriterTest, zero_size_align_inherits_alignment) {
    const std::string weights = "weights";  // 7 bytes
    const std::string tail = "tail";
    const auto tail_tag = hsm::SectionTag::make_device_tag(9, false);

    std::stringstream inherited_stream;
    auto inherited = open_writer(inherited_stream);
    inherited.add_section(hsm::any_device_id, hsm::model_tag, view_of(weights), {/*offset_align=*/64});  // size_align=0
    inherited.add_section(fake_device_id, tail_tag, view_of(tail));
    ASSERT_FALSE(inherited.finalize());

    std::stringstream explicit_stream;
    auto explicit_same = open_writer(explicit_stream);
    explicit_same.add_section(hsm::any_device_id,
                              hsm::model_tag,
                              view_of(weights),
                              {/*offset_align=*/64, /*size_align=*/64});
    explicit_same.add_section(fake_device_id, tail_tag, view_of(tail));
    ASSERT_FALSE(explicit_same.finalize());

    EXPECT_EQ(inherited_stream.str(), explicit_stream.str());
}

// --- add_section() SectionEncoder overloads (sized and unsized) ---

TEST(HsmWriterTest, section_encoder_can_capture_a_copy_of_a_transient_payload) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    {
        const std::string transient = "transient-model";
        writer.add_section(hsm::any_device_id,
                           hsm::model_tag,
                           transient.size(),
                           [bytes = transient](const hsm::SectionSink& sink) {
                               sink({reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()});
                           });
    }  // transient destroyed before finalize() - the encoder already captured its own copy
    ASSERT_FALSE(writer.finalize());

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    const auto* model = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(payload_string(*container, *model), "transient-model");
}

TEST(HsmWriterTest, fill_in_place_writes_generated_payload_into_the_destination) {
    const std::string content = "filled-bytes";

    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id, hsm::model_tag, content.size(), [&content](const hsm::SectionSink& sink) {
        sink({reinterpret_cast<const std::byte*>(content.data()), content.size()});
    });
    ASSERT_FALSE(writer.finalize());

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    const auto* model = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(payload_string(*container, *model), "filled-bytes");
}

TEST(HsmWriterTest, sized_section_encoder_can_split_its_output_into_several_chunks) {
    const std::string expected = "chunk0-chunk1-chunk2-";

    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id, hsm::model_tag, expected.size(), [](const hsm::SectionSink& sink) {
        for (int i = 0; i < 3; ++i) {
            const std::string chunk = "chunk" + std::to_string(i) + "-";
            sink({reinterpret_cast<const std::byte*>(chunk.data()), chunk.size()});
        }
    });
    ASSERT_FALSE(writer.finalize());

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    const auto* model = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(payload_string(*container, *model), expected);
}

TEST(HsmWriterTest, sized_section_encoder_rejects_writing_past_its_declared_size) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id, hsm::model_tag, /*size=*/2, [](const hsm::SectionSink& sink) {
        const std::string data = "too-long";
        sink({reinterpret_cast<const std::byte*>(data.data()), data.size()});
    });
    EXPECT_THROW(writer.finalize(), ov::AssertFailure);
}

TEST(HsmWriterTest, sized_section_encoder_rejects_writing_less_than_its_declared_size) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id, hsm::model_tag, /*size=*/10, [](const hsm::SectionSink& sink) {
        const std::string data = "short";
        sink({reinterpret_cast<const std::byte*>(data.data()), data.size()});
    });
    EXPECT_THROW(writer.finalize(), ov::AssertFailure);
}

TEST(HsmWriterTest, unsized_section_encoder_discovers_size_after_writing) {
    // Simulates content whose length isn't known until it's actually produced (e.g. serializing a
    // variable-length object) - no size is passed to add_section() at all.
    std::stringstream stream;
    auto writer = open_writer(stream);
    writer.add_section(hsm::any_device_id, hsm::model_tag, [](const hsm::SectionSink& sink) {
        for (int i = 0; i < 3; ++i) {
            const std::string chunk = "chunk" + std::to_string(i) + "-";
            sink({reinterpret_cast<const std::byte*>(chunk.data()), chunk.size()});
        }
    });
    ASSERT_FALSE(writer.finalize());

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    const auto* model = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(payload_string(*container, *model), "chunk0-chunk1-chunk2-");
}

TEST(HsmWriterTest, unsized_section_encoder_works_with_a_preallocated_buffer_too) {
    const std::string content = "discovered-at-write-time";

    std::vector<std::byte> buffer(k_buffer_capacity);
    auto writer = open_buffer_writer(buffer.data(), buffer.size());
    writer.add_section(hsm::any_device_id, hsm::model_tag, [&content](const hsm::SectionSink& sink) {
        sink({reinterpret_cast<const std::byte*>(content.data()), content.size()});
    });
    ASSERT_FALSE(writer.finalize());

    const auto container = parse_container(buffer.data(), buffer.size());
    ASSERT_TRUE(container.has_value());
    const auto* model = find_entry(*container, hsm::any_device_id, hsm::model);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(payload_string(*container, *model), "discovered-at-write-time");
}

// --- add_sections() / ISectionWriterHandler ---

TEST(HsmWriterTest, add_sections_lets_a_handler_contribute_its_own_sections) {
    class PluginWriter : public hsm::ISectionWriterHandler {
    public:
        void handle_section(hsm::IWriter& writer) const override {
            writer.add_section(fake_device_id,
                               hsm::SectionTag::make_device_tag(/*local_id=*/3, /*is_inline=*/false),
                               view_of(m_payload));
        }

    private:
        std::string m_payload = "plugin-section";
    };

    std::stringstream stream;
    auto writer = open_writer(stream);
    PluginWriter plugin;
    writer.add_sections({&plugin});
    ASSERT_FALSE(writer.finalize());

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    const auto tag = hsm::SectionTag::make_device_tag(3, false);
    const auto* section = find_entry(*container, fake_device_id, tag.id());
    ASSERT_NE(section, nullptr);
    EXPECT_EQ(payload_string(*container, *section), "plugin-section");
}

TEST(HsmWriterTest, add_sections_skips_a_null_handler) {
    class PluginWriter : public hsm::ISectionWriterHandler {
    public:
        void handle_section(hsm::IWriter& writer) const override {
            writer.add_section(fake_device_id,
                               hsm::SectionTag::make_device_tag(/*local_id=*/4, /*is_inline=*/false),
                               view_of(m_payload));
        }

    private:
        std::string m_payload = "plugin-section";
    };

    std::stringstream stream;
    auto writer = open_writer(stream);
    PluginWriter plugin;
    EXPECT_NO_THROW(writer.add_sections({nullptr, &plugin, nullptr}));
    ASSERT_FALSE(writer.finalize());

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    const auto tag = hsm::SectionTag::make_device_tag(4, false);
    const auto* section = find_entry(*container, fake_device_id, tag.id());
    ASSERT_NE(section, nullptr);
    EXPECT_EQ(payload_string(*container, *section), "plugin-section");
}

TEST(HsmWriterTest, section_encoder_avoids_materializing_a_view_for_a_transient_source_at_all) {
    class PluginWriter : public hsm::ISectionWriterHandler {
    public:
        void handle_section(hsm::IWriter& writer) const override {
            const int value = 42;
            writer.add_section(fake_device_id,
                               hsm::SectionTag::make_device_tag(/*local_id=*/6, /*is_inline=*/false),
                               sizeof(value),
                               [value](const hsm::SectionSink& sink) {
                                   sink({reinterpret_cast<const std::byte*>(&value), sizeof(value)});
                               });
        }
    };

    std::stringstream stream;
    auto writer = open_writer(stream);
    PluginWriter plugin;
    writer.add_sections({&plugin});
    ASSERT_FALSE(writer.finalize());

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    const auto tag = hsm::SectionTag::make_device_tag(6, false);
    const auto* section = find_entry(*container, fake_device_id, tag.id());
    ASSERT_NE(section, nullptr);
    int decoded = 0;
    std::memcpy(&decoded, container->bytes.data() + section->offset, sizeof(decoded));
    EXPECT_EQ(decoded, 42);
}

TEST(HsmWriterTest, handler_based_add_section_reuses_one_handler_for_several_sections_sharing_a_tag) {
    class ShardHandler : public hsm::ISectionWriterHandler {
    public:
        explicit ShardHandler(std::vector<std::string> shards) : m_shards(std::move(shards)) {}

        void handle_section(hsm::IWriter& writer) const override {
            for (size_t i = 0; i < m_shards.size(); ++i) {
                writer.add_section(
                    fake_device_id,
                    hsm::SectionTag::make_device_tag(/*local_id=*/8, /*is_inline=*/false),
                    m_shards[i].size(),
                    [this, i](const hsm::SectionSink& sink) {
                        sink({reinterpret_cast<const std::byte*>(m_shards[i].data()), m_shards[i].size()});
                    });
            }
        }

    private:
        std::vector<std::string> m_shards;
    };

    std::stringstream stream;
    auto writer = open_writer(stream);
    ShardHandler handler({"shard-0", "shard-1"});
    writer.add_sections({&handler});
    ASSERT_FALSE(writer.finalize());

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    const auto tag = hsm::SectionTag::make_device_tag(8, false);
    const auto shards = find_entries(*container, fake_device_id, tag.id());
    ASSERT_EQ(shards.size(), 2u);
    EXPECT_EQ(payload_string(*container, *shards[0]), "shard-0");
    EXPECT_EQ(payload_string(*container, *shards[1]), "shard-1");
}

// --- Wire-format compatibility: the writer's output must satisfy hsm_format.hpp's own contract ---

TEST(HsmWriterTest, written_header_satisfies_the_format_contract) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    const std::string model = "model";
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(model));
    ASSERT_FALSE(writer.finalize());

    hsm::Header header{};
    const auto bytes = stream.str();
    header = hsm::Header::view(reinterpret_cast<const uint8_t*>(bytes.data()));
    EXPECT_TRUE(hsm::is_recognized_header(header));
    EXPECT_TRUE(hsm::is_valid_header_fields(header));
    EXPECT_EQ(header.magic, hsm::BlobMagic::single);
    EXPECT_EQ(header.version_major, hsm::FormatVersion::major);
    EXPECT_EQ(header.version_minor, hsm::FormatVersion::minor);
}

TEST(HsmWriterTest, every_written_manifest_entry_has_valid_section_bounds) {
    std::stringstream stream;
    auto writer = open_writer(stream);
    const std::string id = "id";
    const std::string model = "model-bytes";
    writer.add_section(hsm::any_device_id, hsm::model_id_tag, view_of(id));
    writer.add_section(hsm::any_device_id, hsm::model_tag, view_of(model));
    ASSERT_FALSE(writer.finalize());

    const auto container = parse_container(stream.str());
    ASSERT_TRUE(container.has_value());
    for (const auto& entry : container->entries) {
        EXPECT_TRUE(hsm::is_valid_section_bounds(entry, container->header));
    }
}

// --- WriteErrc / make_error_code ---

TEST(HsmWriterTest, write_errc_message_is_human_readable) {
    const auto ec = hsm::make_error_code(hsm::WriteErrc::write_failed);
    EXPECT_FALSE(ec.message().empty());
    EXPECT_EQ(ec, hsm::WriteErrc::write_failed);
}

TEST(HsmWriterTest, write_errc_category_reports_unknown_for_an_unrecognized_value) {
    const auto ec = hsm::make_error_code(static_cast<hsm::WriteErrc>(0));
    EXPECT_EQ(ec.message(), "unknown error");
    EXPECT_STREQ(ec.category().name(), "ov::runtime::hsm::Write");
}

}  // namespace ov::test
