// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_reader.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace ov::test {
namespace {

enum class FakePluginTag : uint32_t { shard = 0 };

constexpr runtime::DeviceId fake_device_id = 7;

struct EntrySpec {
    runtime::ManifestEntry entry;
    std::vector<uint8_t> payload;
};

runtime::ManifestEntry make_inline_entry(runtime::DeviceId device,
                                         runtime::SectionTag tag,
                                         const std::vector<uint8_t>& bytes) {
    runtime::ManifestEntry entry{};
    entry.device = device;
    entry.tag = tag;
    const auto n = std::min(bytes.size(), entry.inline_bytes.size());
    std::copy(bytes.begin(), bytes.begin() + n, entry.inline_bytes.begin());
    return entry;
}

runtime::ManifestEntry make_pointer_entry(runtime::DeviceId device, runtime::SectionTag tag) {
    runtime::ManifestEntry entry{};
    entry.device = device;
    entry.tag = tag;
    return entry;  // offset/size are filled in by make_container() once payload placement is known.
}

std::vector<uint8_t> make_container(std::vector<EntrySpec> specs,
                                    runtime::BlobMagic magic = runtime::BlobMagic::single) {
    std::vector<uint8_t> payloads;
    for (auto& spec : specs) {
        if (spec.entry.tag.is_pointer()) {
            spec.entry.offset = sizeof(runtime::HSMHeader) + payloads.size();
            spec.entry.size = spec.payload.size();
            payloads.insert(payloads.end(), spec.payload.begin(), spec.payload.end());
        }
    }

    std::vector<uint8_t> manifest(specs.size() * sizeof(runtime::ManifestEntry));
    for (size_t i = 0; i < specs.size(); ++i) {
        std::memcpy(manifest.data() + i * sizeof(runtime::ManifestEntry),
                    &specs[i].entry,
                    sizeof(runtime::ManifestEntry));
    }

    runtime::HSMHeader header{};
    header.magic = magic;
    header.version_major = runtime::HSMFormatVersion::major;
    header.version_minor = runtime::HSMFormatVersion::minor;
    header.manifest_offset = sizeof(runtime::HSMHeader) + payloads.size();
    header.manifest_size = manifest.size();
    header.container_size = header.manifest_offset + header.manifest_size;

    // insert(), not memcpy(data() + offset, ...) - see hsm_format.cpp for why (GCC -Wstringop-overflow).
    std::vector<uint8_t> buffer;
    buffer.reserve(header.container_size);
    const auto* header_bytes = reinterpret_cast<const uint8_t*>(&header);
    buffer.insert(buffer.end(), header_bytes, header_bytes + sizeof(header));
    buffer.insert(buffer.end(), payloads.begin(), payloads.end());
    buffer.insert(buffer.end(), manifest.begin(), manifest.end());
    return buffer;
}

std::vector<uint8_t> bytes_of(const std::string& s) {
    return {s.begin(), s.end()};
}

std::vector<uint8_t> make_sample_reader_container() {
    return make_container({
        {make_inline_entry(runtime::any_device_id, runtime::model_id_tag(), {0xAA, 0xBB, 0xCC, 0xDD}), {}},
        {make_pointer_entry(runtime::any_device_id, runtime::model_tag()), bytes_of("compiled-model-bytes")},
        {make_pointer_entry(fake_device_id, runtime::make_device_tag(/*local_id=*/1, /*is_inline=*/false)),
         bytes_of("device-specific-payload")},
    });
}

std::string to_string(const std::vector<std::byte>& section) {
    return {reinterpret_cast<const char*>(section.data()), section.size()};
}

// Minimal IHsmSectionExtension: recognizes one (device, tag id) pair, records what it was handed.
class RecordingExtension : public runtime::IHsmSectionExtension {
public:
    RecordingExtension(runtime::DeviceId device, uint32_t tag_id) : m_device(device), m_tag_id(tag_id) {}

    bool read_section(const runtime::ManifestEntry& entry, ov::util::MemoryView section) override {
        if (entry.device != m_device || entry.tag.id() != m_tag_id) {
            return false;
        }
        last_payload = {reinterpret_cast<const char*>(section.data()), section.size()};
        ++handled_count;
        return true;
    }

    std::string last_payload;
    size_t handled_count = 0;

private:
    runtime::DeviceId m_device;
    uint32_t m_tag_id;
};

// Wraps `blob` in a seekable stream, preceded by `prefix_size` unrelated bytes (to exercise that the
// container is read relative to the stream's position at open() time, not always from offset 0).
std::stringstream make_stream(const std::vector<uint8_t>& blob, size_t prefix_size = 0) {
    std::stringstream stream;
    stream.write(std::string(prefix_size, '\0').data(), static_cast<std::streamsize>(prefix_size));
    stream.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
    stream.seekg(static_cast<std::streamoff>(prefix_size));
    return stream;
}

}  // namespace

TEST(HsmReaderTest, open_rejects_invalid_buffer) {
    const std::vector<uint8_t> garbage(sizeof(runtime::HSMHeader), 0xFF);
    EXPECT_FALSE(runtime::HsmReader::open(garbage.data(), garbage.size()).has_value());
}

TEST(HsmReaderTest, open_accepts_valid_container) {
    const auto blob = make_container({});
    EXPECT_TRUE(runtime::HsmReader::open(blob.data(), blob.size()).has_value());
}

TEST(HsmReaderTest, open_accepts_shared_context_magic_and_reads_its_sections) {
    const auto blob = make_container(
        {
            {make_inline_entry(runtime::any_device_id, runtime::model_id_tag(), {0xAA}), {}},
            {make_pointer_entry(fake_device_id, runtime::make_device_tag(/*local_id=*/1, /*is_inline=*/false)),
             bytes_of("shared-context-payload")},
        },
        runtime::BlobMagic::multi);
    const auto reader = runtime::HsmReader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    EXPECT_EQ(reader->header().magic, runtime::BlobMagic::multi);
    EXPECT_TRUE(reader->is_shared_context());

    const auto shard = reader->section(fake_device_id, runtime::make_device_tag(1, false).id());
    ASSERT_TRUE(shard.has_value());
    EXPECT_EQ(to_string(*shard), "shared-context-payload");
}

TEST(HsmReaderTest, model_id_section_reads_inline_payload) {
    const auto blob = make_sample_reader_container();
    const auto reader = runtime::HsmReader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section(runtime::model_id);
    ASSERT_TRUE(section.has_value());
    ASSERT_EQ(section->size(), 24u);  // full inline capacity - see ManifestEntry::inline_bytes
    EXPECT_EQ(static_cast<uint8_t>(section->data()[0]), 0xAA);
}

TEST(HsmReaderTest, model_section_reads_pointer_payload) {
    const auto blob = make_sample_reader_container();
    const auto reader = runtime::HsmReader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section(runtime::model);
    ASSERT_TRUE(section.has_value());
    EXPECT_EQ(to_string(*section), "compiled-model-bytes");
}

TEST(HsmReaderTest, common_sections_absent_when_manifest_is_empty) {
    const auto blob = make_container({});
    const auto reader = runtime::HsmReader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    EXPECT_FALSE(reader->section(runtime::model_id).has_value());
    EXPECT_FALSE(reader->section(runtime::model).has_value());
    EXPECT_FALSE(reader->section(runtime::runtime_requirements).has_value());
}

TEST(HsmReaderTest, runtime_requirements_section_reads_opaque_payload) {
    const auto blob = make_container({
        {make_pointer_entry(runtime::any_device_id, runtime::runtime_requirements_tag()), bytes_of("req-v1")},
    });
    const auto reader = runtime::HsmReader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section(runtime::runtime_requirements);
    ASSERT_TRUE(section.has_value());
    EXPECT_EQ(to_string(*section), "req-v1");  // reader hands back raw bytes only, never interprets content
}

TEST(HsmReaderTest, move_constructed_reader_still_reads_sections) {
    const auto blob = make_sample_reader_container();
    auto original = runtime::HsmReader::open(blob.data(), blob.size());
    ASSERT_TRUE(original.has_value());

    const runtime::HsmReader moved(std::move(*original));

    const auto section = moved.section(runtime::model_id);
    ASSERT_TRUE(section.has_value());
    EXPECT_EQ(static_cast<uint8_t>(section->data()[0]), 0xAA);

    // Zero-copy view must still resolve too - buffer_source moved along, not aliased via a stale copy.
    const auto view = moved.section_view(runtime::model);
    ASSERT_TRUE(view.has_value());
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(view->data()), view->size()), "compiled-model-bytes");
}

TEST(HsmReaderTest, move_assignment_replaces_existing_reader) {
    const auto blob_a = make_sample_reader_container();
    auto reader_a = runtime::HsmReader::open(blob_a.data(), blob_a.size());
    ASSERT_TRUE(reader_a.has_value());

    const auto blob_b = make_container({
        {make_inline_entry(runtime::any_device_id, runtime::model_id_tag(), {0xBB}), {}},
    });
    auto reader_b = runtime::HsmReader::open(blob_b.data(), blob_b.size());
    ASSERT_TRUE(reader_b.has_value());

    *reader_a = std::move(*reader_b);

    const auto section = reader_a->section(runtime::model_id);
    ASSERT_TRUE(section.has_value());
    EXPECT_EQ(static_cast<uint8_t>(section->data()[0]), 0xBB);
}

TEST(HsmReaderTest, section_returns_nullopt_for_unknown_device_or_tag) {
    const auto blob = make_sample_reader_container();
    const auto reader = runtime::HsmReader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    EXPECT_FALSE(reader->section(/*device=*/42, runtime::model_id).has_value());
    EXPECT_FALSE(reader->section(runtime::any_device_id, /*tag_id=*/12345).has_value());
}

TEST(HsmReaderTest, section_accepts_any_enum_typed_tag_with_or_without_a_device) {
    const auto blob = make_container({
        {make_inline_entry(runtime::any_device_id, runtime::model_id_tag(), {0xAA}), {}},
        {make_pointer_entry(fake_device_id,
                            runtime::SectionTag::make(static_cast<uint32_t>(FakePluginTag::shard),
                                                      /*is_inline=*/false)),
         bytes_of("plugin-shard")},
    });
    const auto reader = runtime::HsmReader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    // HSMTags, device-less (implies any_device_id) - no manual static_cast at the call site.
    EXPECT_TRUE(reader->section(runtime::HSMTags::model_id).has_value());
    // HSMTags, with an explicit device - same template, still no cast needed.
    EXPECT_TRUE(reader->section(runtime::any_device_id, runtime::HSMTags::model_id).has_value());
    // A completely unrelated enum type (mimicking a plugin's own tag enum) - same template handles it too.
    const auto shard = reader->section(fake_device_id, FakePluginTag::shard);
    ASSERT_TRUE(shard.has_value());
    EXPECT_EQ(to_string(*shard), "plugin-shard");
}

TEST(HsmReaderTest, section_returns_only_the_first_of_several_matching_entries) {
    const auto shard_tag = runtime::make_device_tag(/*local_id=*/2, /*is_inline=*/false);
    const auto blob = make_container({
        {make_pointer_entry(fake_device_id, shard_tag), bytes_of("shard-0")},
        {make_pointer_entry(fake_device_id, shard_tag), bytes_of("shard-1")},
    });
    const auto reader = runtime::HsmReader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section(fake_device_id, shard_tag.id());
    ASSERT_TRUE(section.has_value());
    EXPECT_EQ(to_string(*section), "shard-0");
}

TEST(HsmReaderTest, section_decoder_contract_works_with_section_and_section_view) {
    const auto blob = make_sample_reader_container();
    const auto reader = runtime::HsmReader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    struct ModelData {
        std::string bytes;
    };
    const runtime::SectionDecoder<ModelData> decode_model = [](ov::util::MemoryView section) {
        return std::make_optional(
            ModelData{std::string(reinterpret_cast<const char*>(section.data()), section.size())});
    };

    // Copy-based: decoder gets bytes from section(), same as any other caller.
    const auto bytes = reader->section(runtime::model);
    ASSERT_TRUE(bytes.has_value());
    const auto model = decode_model(ov::util::MemoryView{bytes->data(), bytes->size()});
    ASSERT_TRUE(model.has_value());
    EXPECT_EQ(model->bytes, "compiled-model-bytes");

    // Zero-copy: the exact same decoder works unchanged when a view is available instead.
    const auto view = reader->section_view(runtime::model);
    ASSERT_TRUE(view.has_value());
    const auto model_from_view = decode_model(*view);
    ASSERT_TRUE(model_from_view.has_value());
    EXPECT_EQ(model_from_view->bytes, "compiled-model-bytes");
}

TEST(HsmReaderTest, sections_returns_every_matching_entry_in_manifest_order) {
    // Nothing in the format guarantees a (device, tag) pair is unique - e.g. multiple named/indexed shards.
    const auto shard_tag = runtime::make_device_tag(/*local_id=*/2, /*is_inline=*/false);
    const auto blob = make_container({
        {make_pointer_entry(fake_device_id, shard_tag), bytes_of("shard-0")},
        {make_pointer_entry(fake_device_id, runtime::make_device_tag(3, false)), bytes_of("unrelated")},
        {make_pointer_entry(fake_device_id, shard_tag), bytes_of("shard-1")},
    });
    const auto reader = runtime::HsmReader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto sections = reader->sections(fake_device_id, shard_tag.id());
    ASSERT_EQ(sections.size(), 2u);
    EXPECT_EQ(to_string(sections[0]), "shard-0");
    EXPECT_EQ(to_string(sections[1]), "shard-1");
}

TEST(HsmReaderTest, sections_returns_empty_vector_when_no_entry_matches) {
    const auto blob = make_sample_reader_container();
    const auto reader = runtime::HsmReader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    EXPECT_TRUE(reader->sections(/*device=*/42, runtime::model_id).empty());
}

TEST(HsmReaderTest, read_sections_dispatches_to_matching_extension_and_skips_the_rest) {
    const auto blob = make_sample_reader_container();
    const auto reader = runtime::HsmReader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    RecordingExtension matching(fake_device_id, /*tag_id=*/runtime::core_tag_id_range_end + 1);
    RecordingExtension unrelated(/*device=*/9, /*tag_id=*/999);

    const auto handled = reader->read_sections({&unrelated, &matching});
    EXPECT_EQ(handled, 1u);
    EXPECT_EQ(matching.handled_count, 1u);
    EXPECT_EQ(matching.last_payload, "device-specific-payload");
    EXPECT_EQ(unrelated.handled_count, 0u);
}

TEST(HsmReaderTest, read_sections_can_dispatch_core_owned_entries_when_an_extension_registers_for_them) {
    // Customization point: Core sections are ordinary entries here, not special-cased - a plugin extension
    // may register for (any_device_id, model_id) to override/extend how that common section is handled.
    const auto blob = make_sample_reader_container();
    const auto reader = runtime::HsmReader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    RecordingExtension model_id_override(runtime::any_device_id, runtime::model_id);
    EXPECT_EQ(reader->read_sections({&model_id_override}), 1u);
    EXPECT_EQ(model_id_override.handled_count, 1u);
}

TEST(HsmReaderTest, read_sections_returns_zero_for_empty_manifest) {
    const auto blob = make_container({});
    const auto reader = runtime::HsmReader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());
    EXPECT_EQ(reader->read_sections({}), 0u);
}

// --- Zero-copy _view() siblings: only when opened over an addressable buffer (pointer-mode), always for
// inline-mode entries regardless of source -----------------------------------------------------------

TEST(HsmReaderTest, model_section_view_points_into_the_original_buffer) {
    const auto blob = make_sample_reader_container();
    const auto reader = runtime::HsmReader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section_view(runtime::model);
    ASSERT_TRUE(section.has_value());
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(section->data()), section->size()), "compiled-model-bytes");

    const auto* blob_begin = reinterpret_cast<const std::byte*>(blob.data());
    const auto* blob_end = blob_begin + blob.size();
    EXPECT_GE(section->data(), blob_begin);
    EXPECT_LE(section->data() + section->size(), blob_end);
}

TEST(HsmReaderTest, model_id_section_view_available_even_though_it_is_inline) {
    const auto blob = make_sample_reader_container();
    const auto reader = runtime::HsmReader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section_view(runtime::model_id);
    ASSERT_TRUE(section.has_value());
    ASSERT_EQ(section->size(), 24u);
    EXPECT_EQ(static_cast<uint8_t>(section->data()[0]), 0xAA);
}

TEST(HsmReaderTest, sections_view_returns_zero_copy_views_in_order) {
    const auto shard_tag = runtime::make_device_tag(/*local_id=*/2, /*is_inline=*/false);
    const auto blob = make_container({
        {make_pointer_entry(fake_device_id, shard_tag), bytes_of("shard-0")},
        {make_pointer_entry(fake_device_id, shard_tag), bytes_of("shard-1")},
    });
    const auto reader = runtime::HsmReader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto sections = reader->sections_view(fake_device_id, shard_tag.id());
    ASSERT_EQ(sections.size(), 2u);
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(sections[0].data()), sections[0].size()), "shard-0");
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(sections[1].data()), sections[1].size()), "shard-1");
}

// --- Negative tests (Story 2 DoD: invalid header, invalid manifest, invalid offsets, unsupported
// runtime requirements all reject at HsmReader::open(), before any section can be read) ------------------

TEST(HsmReaderTest, rejects_invalid_header_magic) {
    auto blob = make_container({});
    blob[0] = 'X';  // corrupt magic
    EXPECT_FALSE(runtime::HsmReader::open(blob.data(), blob.size()).has_value());
}

TEST(HsmReaderTest, rejects_invalid_manifest_offset) {
    auto blob = make_sample_reader_container();
    auto header = runtime::HSMHeader::view(blob.data());
    header.manifest_offset = sizeof(runtime::HSMHeader) - 1;  // would start inside the header itself
    std::memcpy(blob.data(), &header, sizeof(header));

    EXPECT_FALSE(runtime::HsmReader::open(blob.data(), blob.size()).has_value());
}

TEST(HsmReaderTest, rejects_invalid_section_offset) {
    auto blob = make_sample_reader_container();
    const auto header = runtime::HSMHeader::view(blob.data());

    // Second manifest entry is the pointer-mode "model" tag; corrupt its offset out of bounds.
    const auto model_entry_offset = header.manifest_offset + sizeof(runtime::ManifestEntry);
    runtime::ManifestEntry entry{};
    std::memcpy(&entry, blob.data() + model_entry_offset, sizeof(entry));
    entry.offset = blob.size() + 1;
    std::memcpy(blob.data() + model_entry_offset, &entry, sizeof(entry));

    EXPECT_FALSE(runtime::HsmReader::open(blob.data(), blob.size()).has_value());
}

TEST(HsmReaderTest, rejects_container_with_out_of_bounds_runtime_requirements) {
    auto blob = make_container({
        {make_pointer_entry(runtime::any_device_id, runtime::runtime_requirements_tag()), bytes_of("req")},
    });
    auto header = runtime::HSMHeader::view(blob.data());
    runtime::ManifestEntry entry{};
    std::memcpy(&entry, blob.data() + header.manifest_offset, sizeof(entry));
    entry.size = std::numeric_limits<runtime::HSMSizeType>::max();  // unsupported: payload can't possibly fit
    std::memcpy(blob.data() + header.manifest_offset, &entry, sizeof(entry));

    EXPECT_FALSE(runtime::HsmReader::open(blob.data(), blob.size()).has_value());
}

// --- HsmReader over a std::istream& source: same class, same output type, just a different open() -------

TEST(HsmReaderStreamTest, open_rejects_invalid_stream) {
    auto blob = make_container({});
    blob[0] = 'X';  // corrupt magic
    auto stream = make_stream(blob);
    EXPECT_FALSE(runtime::HsmReader::open(stream).has_value());
}

TEST(HsmReaderStreamTest, open_accepts_valid_container_at_nonzero_stream_position) {
    const auto blob = make_sample_reader_container();
    auto stream = make_stream(blob, /*prefix_size=*/16);
    const auto reader = runtime::HsmReader::open(stream);
    ASSERT_TRUE(reader.has_value());
    EXPECT_EQ(reader->header().magic, runtime::BlobMagic::single);
}

TEST(HsmReaderStreamTest, open_accepts_shared_context_magic_and_reads_its_sections) {
    const auto blob = make_container(
        {
            {make_pointer_entry(fake_device_id, runtime::make_device_tag(/*local_id=*/1, /*is_inline=*/false)),
             bytes_of("shared-context-payload")},
        },
        runtime::BlobMagic::multi);
    auto stream = make_stream(blob);
    const auto reader = runtime::HsmReader::open(stream);
    ASSERT_TRUE(reader.has_value());

    EXPECT_TRUE(reader->is_shared_context());
    const auto shard = reader->section(fake_device_id, runtime::make_device_tag(1, false).id());
    ASSERT_TRUE(shard.has_value());
    EXPECT_EQ(to_string(*shard), "shared-context-payload");
}

TEST(HsmReaderStreamTest, model_id_and_model_sections_read_correctly) {
    const auto blob = make_sample_reader_container();
    auto stream = make_stream(blob);
    const auto reader = runtime::HsmReader::open(stream);
    ASSERT_TRUE(reader.has_value());

    const auto model_id = reader->section(runtime::model_id);
    ASSERT_TRUE(model_id.has_value());
    ASSERT_EQ(model_id->size(), 24u);
    EXPECT_EQ(static_cast<uint8_t>(model_id->data()[0]), 0xAA);

    const auto model = reader->section(runtime::model);
    ASSERT_TRUE(model.has_value());
    EXPECT_EQ(to_string(*model), "compiled-model-bytes");
}

TEST(HsmReaderStreamTest, move_constructed_reader_still_reads_sections) {
    const auto blob = make_sample_reader_container();
    auto stream = make_stream(blob);
    auto original = runtime::HsmReader::open(stream);
    ASSERT_TRUE(original.has_value());

    const runtime::HsmReader moved(std::move(*original));

    const auto section = moved.section(runtime::model);
    ASSERT_TRUE(section.has_value());
    EXPECT_EQ(to_string(*section), "compiled-model-bytes");
}

TEST(HsmReaderStreamTest, move_assignment_replaces_existing_reader) {
    const auto blob_a = make_sample_reader_container();
    auto stream_a = make_stream(blob_a);
    auto reader_a = runtime::HsmReader::open(stream_a);
    ASSERT_TRUE(reader_a.has_value());

    const auto blob_b = make_container({
        {make_inline_entry(runtime::any_device_id, runtime::model_id_tag(), {0xBB}), {}},
    });
    auto stream_b = make_stream(blob_b);
    auto reader_b = runtime::HsmReader::open(stream_b);
    ASSERT_TRUE(reader_b.has_value());

    *reader_a = std::move(*reader_b);

    const auto section = reader_a->section(runtime::model_id);
    ASSERT_TRUE(section.has_value());
    EXPECT_EQ(static_cast<uint8_t>(section->data()[0]), 0xBB);
}

TEST(HsmReaderStreamTest, view_siblings_work_for_inline_but_not_pointer_mode_entries) {
    const auto blob = make_sample_reader_container();
    auto stream = make_stream(blob);
    const auto reader = runtime::HsmReader::open(stream);
    ASSERT_TRUE(reader.has_value());

    const auto model_id_view = reader->section_view(runtime::model_id);
    ASSERT_TRUE(model_id_view.has_value());
    ASSERT_EQ(model_id_view->size(), 24u);
    EXPECT_EQ(static_cast<uint8_t>(model_id_view->data()[0]), 0xAA);

    EXPECT_FALSE(reader->section_view(runtime::model).has_value());
}

TEST(HsmReaderStreamTest, common_sections_absent_when_manifest_is_empty) {
    const auto blob = make_container({});
    auto stream = make_stream(blob);
    const auto reader = runtime::HsmReader::open(stream);
    ASSERT_TRUE(reader.has_value());

    EXPECT_FALSE(reader->section(runtime::model_id).has_value());
    EXPECT_FALSE(reader->section(runtime::model).has_value());
    EXPECT_FALSE(reader->section(runtime::runtime_requirements).has_value());
}

TEST(HsmReaderStreamTest, sections_returns_every_matching_entry) {
    const auto shard_tag = runtime::make_device_tag(/*local_id=*/2, /*is_inline=*/false);
    const auto blob = make_container({
        {make_pointer_entry(fake_device_id, shard_tag), bytes_of("shard-0")},
        {make_pointer_entry(fake_device_id, shard_tag), bytes_of("shard-1")},
    });
    auto stream = make_stream(blob);
    const auto reader = runtime::HsmReader::open(stream);
    ASSERT_TRUE(reader.has_value());

    const auto sections = reader->sections(fake_device_id, shard_tag.id());
    ASSERT_EQ(sections.size(), 2u);
    EXPECT_EQ(to_string(sections[0]), "shard-0");
    EXPECT_EQ(to_string(sections[1]), "shard-1");
}

TEST(HsmReaderStreamTest, read_sections_dispatches_same_extension_type_as_memory_backed_reader) {
    const auto blob = make_sample_reader_container();
    auto stream = make_stream(blob);
    const auto reader = runtime::HsmReader::open(stream);
    ASSERT_TRUE(reader.has_value());

    RecordingExtension extension(fake_device_id, /*tag_id=*/runtime::core_tag_id_range_end + 1);
    EXPECT_EQ(reader->read_sections({&extension}), 1u);
    EXPECT_EQ(extension.last_payload, "device-specific-payload");
}

TEST(HsmReaderStreamTest, rejects_invalid_manifest_offset) {
    auto blob = make_sample_reader_container();
    auto header = runtime::HSMHeader::view(blob.data());
    header.manifest_offset = sizeof(runtime::HSMHeader) - 1;  // would start inside the header itself
    header.container_size = blob.size();  // ensure container_size is consistent with the actual blob size
    std::memcpy(blob.data(), &header, sizeof(header));

    auto stream = make_stream(blob);
    EXPECT_FALSE(runtime::HsmReader::open(stream).has_value());
}

}  // namespace ov::test
