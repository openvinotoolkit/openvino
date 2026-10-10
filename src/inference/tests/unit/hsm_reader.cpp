// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_reader.hpp"

#include <gmock/gmock.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace ov::test {
namespace hsm = ov::runtime::hsm;
namespace {

enum class FakePluginTag : uint32_t { shard = 0 };

constexpr hsm::DeviceId fake_device_id = 7;

struct EntrySpec {
    hsm::ManifestEntry entry;
    std::string payload;
};

hsm::ManifestEntry make_inline_entry(hsm::DeviceId device, hsm::SectionTag tag, const std::vector<uint8_t>& bytes) {
    hsm::ManifestEntry entry{};
    entry.device = device;
    entry.tag = tag;
    const auto n = std::min(bytes.size(), entry.inline_bytes.size());
    std::copy(bytes.begin(), bytes.begin() + n, entry.inline_bytes.begin());
    return entry;
}

hsm::ManifestEntry make_pointer_entry(hsm::DeviceId device, hsm::SectionTag tag) {
    hsm::ManifestEntry entry{};
    entry.device = device;
    entry.tag = tag;
    return entry;  // offset/size are filled in by make_container() once payload placement is known.
}

// The one place that works around GCC 11's -Wstringop-overread false positive on vector::insert(it, begin, end)
template <typename Container>
void append(std::vector<uint8_t>& dst, const Container& src) {
    if (!src.empty()) {
        const auto offset = dst.size();
        dst.resize(offset + src.size());
        std::memcpy(dst.data() + offset, src.data(), src.size());
    }
}

std::vector<uint8_t> make_container(std::vector<EntrySpec> specs, hsm::BlobMagic magic = hsm::BlobMagic::single) {
    std::vector<uint8_t> payloads;
    for (auto& spec : specs) {
        if (spec.entry.tag.is_pointer()) {
            spec.entry.offset = sizeof(hsm::Header) + payloads.size();
            spec.entry.size = spec.payload.size();
            append(payloads, spec.payload);
        }
    }

    std::vector<uint8_t> manifest(specs.size() * sizeof(hsm::ManifestEntry));
    for (size_t i = 0; i < specs.size(); ++i) {
        std::memcpy(manifest.data() + i * sizeof(hsm::ManifestEntry), &specs[i].entry, sizeof(hsm::ManifestEntry));
    }

    hsm::Header header{};
    header.magic = magic;
    header.version_major = hsm::FormatVersion::major;
    header.version_minor = hsm::FormatVersion::minor;
    header.manifest_offset = sizeof(hsm::Header) + payloads.size();
    header.manifest_size = manifest.size();
    header.container_size = header.manifest_offset + header.manifest_size;

    std::vector<uint8_t> buffer(sizeof(header));
    std::memcpy(buffer.data(), &header, sizeof(header));
    append(buffer, payloads);
    append(buffer, manifest);
    return buffer;
}

std::string to_string(const std::vector<std::byte>& section) {
    return {reinterpret_cast<const char*>(section.data()), section.size()};
}

std::string to_string(const hsm::Section& section) {
    const auto bytes = section.to_bytes();
    return bytes ? to_string(*bytes) : std::string{};
}

std::vector<uint8_t> make_sample_reader_container() {
    return make_container({
        {make_inline_entry(hsm::any_device_id, hsm::model_id_tag, {0xAA, 0xBB, 0xCC, 0xDD}), {}},
        {make_pointer_entry(hsm::any_device_id, hsm::model_tag), "compiled-model-bytes"},
        {make_pointer_entry(fake_device_id, hsm::SectionTag::make_device_tag(/*local_id=*/1, /*is_inline=*/false)),
         "device-specific-payload"},
    });
}

// Minimal ISectionReaderHandler: recognizes one (device, tag id) pair, records what it was handed.
class RecordingHandler : public hsm::ISectionReaderHandler {
public:
    RecordingHandler(hsm::DeviceId device, uint32_t tag_id) : m_device(device), m_tag_id(tag_id) {}

    bool handle_section(const hsm::ManifestEntry& entry, ov::util::MemoryView section) override {
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
    hsm::DeviceId m_device;
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

class MockStreamBuf : public std::streambuf {
public:
    explicit MockStreamBuf(const std::vector<uint8_t>& data) : m_real(std::string(data.begin(), data.end())) {
        ON_CALL(*this, xsgetn).WillByDefault([this](char_type* dst, std::streamsize n) {
            return m_real.sgetn(dst, n);
        });
        ON_CALL(*this, seekoff)
            .WillByDefault([this](off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) {
                return m_real.pubseekoff(off, dir, which);
            });
        ON_CALL(*this, seekpos).WillByDefault([this](pos_type pos, std::ios_base::openmode which) {
            return m_real.pubseekpos(pos, which);
        });
        ON_CALL(*this, underflow).WillByDefault([this]() {
            return m_real.sgetc();
        });
        ON_CALL(*this, uflow).WillByDefault([this]() {
            return m_real.sbumpc();
        });
    }

    MOCK_METHOD(std::streamsize, xsgetn, (char_type * dst, std::streamsize n), (override));
    MOCK_METHOD(pos_type,
                seekoff,
                (off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which),
                (override));
    MOCK_METHOD(pos_type, seekpos, (pos_type pos, std::ios_base::openmode which), (override));
    MOCK_METHOD(int_type, underflow, (), (override));
    MOCK_METHOD(int_type, uflow, (), (override));

private:
    std::stringbuf m_real;
};

}  // namespace

// --- Section: constructed directly, without a Reader - it's a publicly exposed type on its own ----------

TEST(HsmSectionTest, inline_mode_reads_directly_from_the_entry_no_source_needed) {
    const auto entry = make_inline_entry(fake_device_id, hsm::model_id_tag, {0xAA, 0xBB, 0xCC});
    const hsm::Section section(entry);

    EXPECT_EQ(section.device(), fake_device_id);
    EXPECT_EQ(section.tag().id(), hsm::model_id);
    EXPECT_EQ(section.size(), entry.inline_bytes.size());

    std::vector<std::byte> destination(section.size());
    ASSERT_TRUE(section.read(destination.data()));
    EXPECT_EQ(destination[0], std::byte{0xAA});
}

TEST(HsmSectionTest, pointer_mode_reads_from_an_explicitly_supplied_buffer) {
    const std::string payload = "standalone-section-bytes";
    auto entry = make_pointer_entry(fake_device_id, hsm::model_tag);
    entry.size = payload.size();
    const ov::util::MemoryView view{reinterpret_cast<const std::byte*>(payload.data()), payload.size()};
    const hsm::Section section(entry, view);

    const auto direct = section.view();
    ASSERT_TRUE(direct.has_value());
    EXPECT_EQ(to_string(std::vector<std::byte>(direct->begin(), direct->end())), payload);
}

TEST(HsmSectionTest, read_rejects_a_null_destination) {
    const auto entry = make_inline_entry(fake_device_id, hsm::model_id_tag, {0xAA});
    const hsm::Section section(entry);
    EXPECT_FALSE(section.read(0, section.size(), nullptr));
}

TEST(HsmSectionTest, pointer_mode_entry_built_with_the_inline_only_constructor_has_no_source) {
    // Misuses the public API directly: the inline-only constructor never sets a payload source, so a
    // pointer-mode entry built this way has nothing for view()/read() to fall back on (m_source stays
    // std::monostate).
    auto entry = make_pointer_entry(fake_device_id, hsm::model_tag);
    entry.size = 4;
    const hsm::Section section(entry);

    EXPECT_FALSE(section.view().has_value());
    std::vector<std::byte> destination(4);
    EXPECT_FALSE(section.read(destination.data()));
    EXPECT_FALSE(section.to_bytes().has_value());
}

// --- ISectionReaderHandler: constructed directly, without a Reader driving dispatch ---------------------

TEST(ISectionReaderHandlerTest, recognizes_own_device_and_tag) {
    hsm::ManifestEntry entry{};
    entry.device = fake_device_id;
    entry.tag = hsm::SectionTag::make(/*id=*/42, /*is_inline=*/false);

    const std::byte payload[4] = {};
    RecordingHandler handler(fake_device_id, /*tag_id=*/42);
    EXPECT_TRUE(handler.handle_section(entry, ov::util::MemoryView{payload, 4}));
    EXPECT_EQ(handler.last_payload.size(), 4u);
    EXPECT_EQ(handler.handled_count, 1u);
}

TEST(ISectionReaderHandlerTest, skips_entry_it_does_not_own) {
    hsm::ManifestEntry entry{};
    entry.device = fake_device_id + 1;  // different device
    entry.tag = hsm::SectionTag::make(/*id=*/42, /*is_inline=*/false);

    RecordingHandler handler(fake_device_id, /*tag_id=*/42);
    EXPECT_FALSE(handler.handle_section(entry, ov::util::MemoryView{}));
    EXPECT_EQ(handler.handled_count, 0u);  // never called
}

TEST(HsmReaderTest, open_rejects_invalid_buffer) {
    const std::vector<uint8_t> garbage(sizeof(hsm::Header), 0xFF);
    EXPECT_FALSE(hsm::Reader::open(garbage.data(), garbage.size()).has_value());
}

TEST(HsmReaderTest, open_rejects_a_buffer_smaller_than_the_header_itself) {
    const std::vector<uint8_t> too_small(sizeof(hsm::Header) - 1, 0xFF);
    EXPECT_FALSE(hsm::Reader::open(too_small.data(), too_small.size()).has_value());
}

TEST(HsmReaderTest, open_accepts_valid_container) {
    const auto blob = make_container({});
    EXPECT_TRUE(hsm::Reader::open(blob.data(), blob.size()).has_value());
}

TEST(HsmReaderTest, open_accepts_shared_context_magic_and_reads_its_sections) {
    const auto blob = make_container(
        {
            {make_inline_entry(hsm::any_device_id, hsm::model_id_tag, {0xAA}), {}},
            {make_pointer_entry(fake_device_id, hsm::SectionTag::make_device_tag(/*local_id=*/1, /*is_inline=*/false)),
             "shared-context-payload"},
        },
        hsm::BlobMagic::multi);
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto shard = reader->section(fake_device_id, hsm::SectionTag::make_device_tag(1, false).id());
    ASSERT_TRUE(shard.has_value());
    EXPECT_EQ(to_string(*shard), "shared-context-payload");
}

TEST(HsmReaderTest, model_id_section_reads_inline_payload) {
    const auto blob = make_sample_reader_container();
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section(hsm::model_id);
    ASSERT_TRUE(section.has_value());
    ASSERT_EQ(section->size(), 24u);  // full inline capacity - see ManifestEntry::inline_bytes
    const auto view = section->view();
    ASSERT_TRUE(view.has_value());  // inline is always viewable, regardless of source
    EXPECT_EQ(static_cast<uint8_t>(view->data()[0]), 0xAA);
}

TEST(HsmReaderTest, model_section_reads_pointer_payload) {
    const auto blob = make_sample_reader_container();
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section(hsm::model);
    ASSERT_TRUE(section.has_value());
    EXPECT_EQ(to_string(*section), "compiled-model-bytes");
}

TEST(HsmReaderTest, common_sections_absent_when_manifest_is_empty) {
    const auto blob = make_container({});
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    EXPECT_FALSE(reader->section(hsm::model_id).has_value());
    EXPECT_FALSE(reader->section(hsm::model).has_value());
    EXPECT_FALSE(reader->section(hsm::runtime_requirements).has_value());
}

TEST(HsmReaderTest, runtime_requirements_section_reads_opaque_payload) {
    const auto blob = make_container({
        {make_pointer_entry(hsm::any_device_id, hsm::runtime_requirements_tag), "req-v1"},
    });
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section(hsm::runtime_requirements);
    ASSERT_TRUE(section.has_value());
    EXPECT_EQ(to_string(*section), "req-v1");  // reader hands back raw bytes only, never interprets content
}

TEST(HsmReaderTest, move_constructed_reader_still_reads_sections) {
    const auto blob = make_sample_reader_container();
    auto original = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(original.has_value());

    const hsm::Reader moved(std::move(*original));

    // Zero-copy view must still resolve too - the buffer moved along, not aliased via a stale copy.
    const auto section = moved.section(hsm::model);
    ASSERT_TRUE(section.has_value());
    const auto view = section->view();
    ASSERT_TRUE(view.has_value());
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(view->data()), view->size()), "compiled-model-bytes");
}

TEST(HsmReaderTest, move_assignment_replaces_existing_reader) {
    const auto blob_a = make_sample_reader_container();
    auto reader_a = hsm::Reader::open(blob_a.data(), blob_a.size());
    ASSERT_TRUE(reader_a.has_value());

    const auto blob_b = make_container({
        {make_inline_entry(hsm::any_device_id, hsm::model_id_tag, {0xBB}), {}},
    });
    auto reader_b = hsm::Reader::open(blob_b.data(), blob_b.size());
    ASSERT_TRUE(reader_b.has_value());

    *reader_a = std::move(*reader_b);

    const auto section = reader_a->section(hsm::model_id);
    ASSERT_TRUE(section.has_value());
    const auto view = section->view();
    ASSERT_TRUE(view.has_value());
    EXPECT_EQ(static_cast<uint8_t>(view->data()[0]), 0xBB);
}

TEST(HsmReaderTest, section_returns_nullopt_for_unknown_device_or_tag) {
    const auto blob = make_sample_reader_container();
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    EXPECT_FALSE(reader->section(/*device=*/42, hsm::model_id).has_value());
    EXPECT_FALSE(reader->section(hsm::any_device_id, /*tag_id=*/12345).has_value());
}

TEST(HsmReaderTest, section_accepts_any_enum_typed_tag_with_or_without_a_device) {
    const auto blob = make_container({
        {make_inline_entry(hsm::any_device_id, hsm::model_id_tag, {0xAA}), {}},
        {make_pointer_entry(fake_device_id,
                            hsm::SectionTag::make(static_cast<uint32_t>(FakePluginTag::shard),
                                                  /*is_inline=*/false)),
         "plugin-shard"},
    });
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    // Tags, device-less (implies any_device_id) - no manual static_cast at the call site.
    EXPECT_TRUE(reader->section(hsm::Tags::model_id).has_value());
    // Tags, with an explicit device - same template, still no cast needed.
    EXPECT_TRUE(reader->section(hsm::any_device_id, hsm::Tags::model_id).has_value());
    // A completely unrelated enum type (mimicking a plugin's own tag enum) - same template handles it too.
    const auto shard = reader->section(fake_device_id, FakePluginTag::shard);
    ASSERT_TRUE(shard.has_value());
    EXPECT_EQ(to_string(*shard), "plugin-shard");
}

TEST(HsmReaderTest, device_overrides_core_tag_by_reusing_it_under_its_own_device_id) {
    // The device-id override convention - see the note on hsm::DeviceId: a device needing content different
    // from a Core tag's shared meaning reuses that tag id under its own id instead of inventing a new tag.
    const auto blob = make_container({
        {make_pointer_entry(hsm::any_device_id, hsm::compiled_options_tag), "shared-options"},
        {make_pointer_entry(hsm::npu_device_id, hsm::compiled_options_tag), "npu-options"},
    });
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    // One tag id, two entries - they stay distinct because lookup is always by (device, tag), never tag alone.
    const auto shared = reader->section(hsm::Tags::compiled_options);
    ASSERT_TRUE(shared.has_value());
    EXPECT_EQ(to_string(*shared), "shared-options");

    const auto overridden = reader->section(hsm::npu_device_id, hsm::Tags::compiled_options);
    ASSERT_TRUE(overridden.has_value());
    EXPECT_EQ(to_string(*overridden), "npu-options");

    EXPECT_EQ(reader->count(hsm::Tags::compiled_options), 1U);
    EXPECT_EQ(reader->count(hsm::npu_device_id, hsm::Tags::compiled_options), 1U);

    // A device that published no override of its own finds nothing under its id, and so falls back to the
    // shared any_device_id entry - the resolution order every such lookup follows.
    EXPECT_FALSE(reader->section(hsm::gpu_device_id, hsm::Tags::compiled_options).has_value());
}

TEST(HsmReaderTest, section_returns_only_the_first_of_several_matching_entries) {
    const auto shard_tag = hsm::SectionTag::make_device_tag(/*local_id=*/2, /*is_inline=*/false);
    const auto blob = make_container({
        {make_pointer_entry(fake_device_id, shard_tag), "shard-0"},
        {make_pointer_entry(fake_device_id, shard_tag), "shard-1"},
    });
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section(fake_device_id, shard_tag.id());
    ASSERT_TRUE(section.has_value());
    EXPECT_EQ(to_string(*section), "shard-0");
}

TEST(HsmReaderTest, sections_returns_every_matching_entry_in_manifest_order) {
    // Nothing in the format guarantees a (device, tag) pair is unique - e.g. multiple named/indexed shards.
    const auto shard_tag = hsm::SectionTag::make_device_tag(/*local_id=*/2, /*is_inline=*/false);
    const auto blob = make_container({
        {make_pointer_entry(fake_device_id, shard_tag), "shard-0"},
        {make_pointer_entry(fake_device_id, hsm::SectionTag::make_device_tag(3, false)), "unrelated"},
        {make_pointer_entry(fake_device_id, shard_tag), "shard-1"},
    });
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto sections = reader->sections(fake_device_id, shard_tag.id());
    ASSERT_EQ(sections.size(), 2u);
    EXPECT_EQ(to_string(sections[0]), "shard-0");
    EXPECT_EQ(to_string(sections[1]), "shard-1");
}

TEST(HsmReaderTest, sections_returns_empty_vector_when_no_entry_matches) {
    const auto blob = make_sample_reader_container();
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    EXPECT_TRUE(reader->sections(/*device=*/42, hsm::model_id).empty());
}

TEST(HsmReaderTest, sections_any_device_id_overload_matches_the_explicit_any_device_id_form) {
    const auto blob = make_sample_reader_container();
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto sections = reader->sections(hsm::model);
    ASSERT_EQ(sections.size(), 1u);
    EXPECT_EQ(to_string(sections[0]), "compiled-model-bytes");
    // fake_device_id-owned tag must not surface through the any_device_id-only overload.
    EXPECT_TRUE(reader->sections(hsm::SectionTag::make_device_tag(/*local_id=*/1, /*is_inline=*/false).id()).empty());
}

TEST(HsmReaderTest, count_matches_sections_size_without_reading_any_payload) {
    const auto shard_tag = hsm::SectionTag::make_device_tag(/*local_id=*/2, /*is_inline=*/false);
    const auto blob = make_container({
        {make_pointer_entry(fake_device_id, shard_tag), "shard-0"},
        {make_pointer_entry(fake_device_id, hsm::SectionTag::make_device_tag(3, false)), "unrelated"},
        {make_pointer_entry(fake_device_id, shard_tag), "shard-1"},
    });
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    EXPECT_EQ(reader->count(fake_device_id, shard_tag.id()), 2u);
    EXPECT_EQ(reader->count(/*device=*/42, hsm::model_id), 0u);
}

TEST(HsmReaderTest, count_any_device_id_overload_matches_the_explicit_any_device_id_form) {
    const auto blob = make_sample_reader_container();
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    // model is any_device_id-owned in make_sample_reader_container() - the any_device_id-only overload
    // must find it too, not just count(any_device_id, tag).
    EXPECT_EQ(reader->count(hsm::model), 1u);
    // fake_device_id owns a different tag - the any_device_id-only overload must not see it.
    EXPECT_EQ(reader->count(hsm::SectionTag::make_device_tag(/*local_id=*/1, /*is_inline=*/false).id()), 0u);
}

TEST(HsmReaderTest, entries_gives_a_manifest_overview_without_reading_any_payload) {
    const auto blob = make_sample_reader_container();
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    ASSERT_EQ(reader->entries().size(), 3u);
    EXPECT_EQ(reader->entries()[0].tag.id(), hsm::model_id);
    EXPECT_EQ(reader->entries()[1].tag.id(), hsm::model);
    EXPECT_EQ(reader->entries()[2].device, fake_device_id);
}

TEST(HsmReaderTest, read_sections_dispatches_to_matching_handler_and_skips_the_rest) {
    const auto blob = make_sample_reader_container();
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    RecordingHandler matching(fake_device_id, /*tag_id=*/hsm::core_tag_id_range_end + 1);
    RecordingHandler unrelated(/*device=*/9, /*tag_id=*/999);

    const auto handled = reader->read_sections({&unrelated, &matching});
    EXPECT_EQ(handled, 1u);
    EXPECT_EQ(matching.handled_count, 1u);
    EXPECT_EQ(matching.last_payload, "device-specific-payload");
    EXPECT_EQ(unrelated.handled_count, 0u);
}

TEST(HsmReaderTest, read_sections_can_dispatch_core_owned_entries_when_a_handler_registers_for_them) {
    // Customization point: Core sections are ordinary entries here, not special-cased - a plugin extension
    // may register for (any_device_id, model_id) to override/extend how that common section is handled.
    const auto blob = make_sample_reader_container();
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    RecordingHandler model_id_override(hsm::any_device_id, hsm::model_id);
    EXPECT_EQ(reader->read_sections({&model_id_override}), 1u);
    EXPECT_EQ(model_id_override.handled_count, 1u);
}

TEST(HsmReaderTest, read_sections_returns_zero_for_empty_manifest) {
    const auto blob = make_container({});
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());
    EXPECT_EQ(reader->read_sections({}), 0u);
}

// --- Section::view(): zero-copy, only when opened over an addressable buffer (pointer-mode), always for
// inline-mode entries regardless of source -----------------------------------------------------------

TEST(HsmReaderTest, section_view_points_into_the_original_buffer) {
    const auto blob = make_sample_reader_container();
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section(hsm::model);
    ASSERT_TRUE(section.has_value());
    const auto view = section->view();
    ASSERT_TRUE(view.has_value());
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(view->data()), view->size()), "compiled-model-bytes");

    const auto* blob_begin = reinterpret_cast<const std::byte*>(blob.data());
    const auto* blob_end = blob_begin + blob.size();
    EXPECT_GE(view->data(), blob_begin);
    EXPECT_LE(view->data() + view->size(), blob_end);
}

TEST(HsmReaderTest, section_view_available_even_for_inline_entries) {
    const auto blob = make_sample_reader_container();
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section(hsm::model_id);
    ASSERT_TRUE(section.has_value());
    const auto view = section->view();
    ASSERT_TRUE(view.has_value());
    ASSERT_EQ(view->size(), 24u);
    EXPECT_EQ(static_cast<uint8_t>(view->data()[0]), 0xAA);
}

// --- Section::read(): section()/view() are whole-section only; this is for reading arbitrary ranges one
// at a time, deciding what's next from what a previous call already returned --------------------------

TEST(HsmReaderTest, section_read_reads_a_slice_of_a_pointer_mode_section) {
    const std::string payload = "compiled-model-bytes";
    const auto blob = make_sample_reader_container();
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section(hsm::model);
    ASSERT_TRUE(section.has_value());

    std::vector<std::byte> destination(5);
    ASSERT_TRUE(section->read(9, 5, destination.data()));
    EXPECT_EQ(to_string(destination), payload.substr(9, 5));
}

TEST(HsmReaderTest, section_read_rejects_a_window_exceeding_the_section_size) {
    const auto blob = make_sample_reader_container();
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section(hsm::model);
    ASSERT_TRUE(section.has_value());

    std::vector<std::byte> destination(section->size());
    EXPECT_FALSE(section->read(1, section->size(), destination.data()));
    EXPECT_FALSE(section->read(section->size() + 1, 0, destination.data()));
}

TEST(HsmReaderTest, section_read_supports_several_calls_with_the_same_decoder) {
    const std::string payload = "compiled-model-bytes";
    const auto blob = make_sample_reader_container();
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section(hsm::model);
    ASSERT_TRUE(section.has_value());

    struct Piece {
        std::string bytes;
    };
    const hsm::SectionDecoder<Piece> decode_piece = [](const hsm::Section& s) -> std::optional<Piece> {
        // Same decoder shape whether it's given the whole section (via view()/read()) or, as here, asked
        // to decode only a range read into a caller-owned buffer.
        std::vector<std::byte> bytes(s.size());
        return s.read(bytes.data()) ? std::make_optional(Piece{to_string(bytes)}) : std::nullopt;
    };

    // Simulates deciding what to read next from what was already decoded - same decoder, two ranges.
    std::vector<std::byte> head_bytes(9);
    ASSERT_TRUE(section->read(0, 9, head_bytes.data()));
    EXPECT_EQ(to_string(head_bytes), payload.substr(0, 9));

    std::vector<std::byte> tail_bytes(5);
    ASSERT_TRUE(section->read(9, 5, tail_bytes.data()));
    EXPECT_EQ(to_string(tail_bytes), payload.substr(9, 5));

    // The same decoder also still works when run through the reader, over the whole section, unchanged.
    const auto whole = reader->decode(hsm::model, decode_piece);
    ASSERT_TRUE(whole.has_value());
    EXPECT_EQ(whole->bytes, payload);
}

TEST(HsmReaderTest, section_read_works_for_inline_entries) {
    const auto blob = make_sample_reader_container();
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section(hsm::model_id);
    ASSERT_TRUE(section.has_value());

    std::byte first_byte{};
    ASSERT_TRUE(section->read(0, 1, &first_byte));
    EXPECT_EQ(static_cast<uint8_t>(first_byte), 0xAA);
}

// --- decode()/decode_all(): the same SectionDecoder<T> works whether the reader is buffer- or
// stream-backed - see HsmReaderStreamTest.decode_falls_back_to_read_when_no_view_is_available -----------

TEST(HsmReaderTest, decode_uses_the_zero_copy_view_when_one_is_available) {
    const auto blob = make_sample_reader_container();
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const auto* blob_begin = reinterpret_cast<const std::byte*>(blob.data());
    const auto* blob_end = blob_begin + blob.size();
    // Exercises make_section_decoder()'s zero-copy path: parse receives the section's own view
    // unchanged, still pointing inside the original buffer - not a copy routed through to_bytes().
    const auto decode_as_view = hsm::make_section_decoder<ov::util::MemoryView>([](ov::util::MemoryView view) {
        return std::make_optional(view);
    });

    const auto decoded = reader->decode(hsm::model, decode_as_view);
    ASSERT_TRUE(decoded.has_value());
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(decoded->data()), decoded->size()), "compiled-model-bytes");
    EXPECT_GE(decoded->data(), blob_begin);
    EXPECT_LE(decoded->data() + decoded->size(), blob_end);
}

TEST(HsmReaderTest, decode_returns_nullopt_for_unknown_tag) {
    const auto blob = make_sample_reader_container();
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const hsm::SectionDecoder<int> decode_length = [](const hsm::Section& section) {
        return std::make_optional(static_cast<int>(section.size()));
    };
    EXPECT_FALSE(reader->decode(/*device=*/42, hsm::model_id, decode_length).has_value());
}

TEST(HsmReaderTest, decode_with_an_explicit_device_finds_a_device_owned_section) {
    const auto blob = make_sample_reader_container();
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const hsm::SectionDecoder<std::string> decode_bytes = [](const hsm::Section& section) {
        const auto view = section.view();
        return view ? std::make_optional(std::string(reinterpret_cast<const char*>(view->data()), view->size()))
                    : std::nullopt;
    };

    const auto decoded = reader->decode(fake_device_id,
                                        hsm::SectionTag::make_device_tag(/*local_id=*/1, /*is_inline=*/false).id(),
                                        decode_bytes);
    ASSERT_TRUE(decoded.has_value());
    EXPECT_EQ(*decoded, "device-specific-payload");
}

TEST(HsmReaderTest, decode_all_returns_every_matching_entry_decoded_in_order) {
    const auto shard_tag = hsm::SectionTag::make_device_tag(/*local_id=*/2, /*is_inline=*/false);
    const auto blob = make_container({
        {make_pointer_entry(fake_device_id, shard_tag), "shard-0"},
        {make_pointer_entry(fake_device_id, shard_tag), "shard-1"},
    });
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const hsm::SectionDecoder<std::string> decode_shard = [](const hsm::Section& section) {
        const auto view = section.view();
        return view ? std::make_optional(std::string(reinterpret_cast<const char*>(view->data()), view->size()))
                    : std::nullopt;
    };

    const auto shards = reader->decode_all(fake_device_id, shard_tag.id(), decode_shard);
    ASSERT_EQ(shards.size(), 2u);
    EXPECT_EQ(shards[0], "shard-0");
    EXPECT_EQ(shards[1], "shard-1");
}

TEST(HsmReaderTest, decode_all_returns_an_empty_vector_when_nothing_matches) {
    const auto blob = make_sample_reader_container();
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const hsm::SectionDecoder<std::string> decode_bytes = [](const hsm::Section&) {
        return std::make_optional(std::string{});
    };

    EXPECT_TRUE(reader->decode_all(/*device=*/42, hsm::model_id, decode_bytes).empty());
}

TEST(HsmReaderTest, decode_all_any_device_id_overload_matches_the_explicit_any_device_id_form) {
    const auto blob = make_sample_reader_container();
    const auto reader = hsm::Reader::open(blob.data(), blob.size());
    ASSERT_TRUE(reader.has_value());

    const hsm::SectionDecoder<std::string> decode_bytes = [](const hsm::Section& section) {
        const auto view = section.view();
        return view ? std::make_optional(std::string(reinterpret_cast<const char*>(view->data()), view->size()))
                    : std::nullopt;
    };

    const auto decoded = reader->decode_all(hsm::model, decode_bytes);
    ASSERT_EQ(decoded.size(), 1u);
    EXPECT_EQ(decoded[0], "compiled-model-bytes");
}

// --- Negative tests (Story 2 DoD: invalid header, invalid manifest, invalid offsets, unsupported
// runtime requirements all reject at Reader::open(), before any section can be read) ------------------

TEST(HsmReaderTest, rejects_invalid_header_magic) {
    auto blob = make_container({});
    blob[0] = 'X';  // corrupt magic
    EXPECT_FALSE(hsm::Reader::open(blob.data(), blob.size()).has_value());
}

TEST(HsmReaderTest, rejects_invalid_manifest_offset) {
    auto blob = make_sample_reader_container();
    auto header = hsm::Header::view(blob.data());
    header.manifest_offset = sizeof(hsm::Header) - 1;  // would start inside the header itself
    std::memcpy(blob.data(), &header, sizeof(header));

    EXPECT_FALSE(hsm::Reader::open(blob.data(), blob.size()).has_value());
}

TEST(HsmReaderTest, rejects_invalid_section_offset) {
    auto blob = make_sample_reader_container();
    const auto header = hsm::Header::view(blob.data());

    // Second manifest entry is the pointer-mode "model" tag; corrupt its offset out of bounds.
    const auto model_entry_offset = header.manifest_offset + sizeof(hsm::ManifestEntry);
    hsm::ManifestEntry entry{};
    std::memcpy(&entry, blob.data() + model_entry_offset, sizeof(entry));
    entry.offset = blob.size() + 1;
    std::memcpy(blob.data() + model_entry_offset, &entry, sizeof(entry));

    EXPECT_FALSE(hsm::Reader::open(blob.data(), blob.size()).has_value());
}

TEST(HsmReaderTest, rejects_container_with_out_of_bounds_runtime_requirements) {
    auto blob = make_container({
        {make_pointer_entry(hsm::any_device_id, hsm::runtime_requirements_tag), "req"},
    });
    auto header = hsm::Header::view(blob.data());
    hsm::ManifestEntry entry{};
    std::memcpy(&entry, blob.data() + header.manifest_offset, sizeof(entry));
    entry.size = std::numeric_limits<hsm::SizeType>::max();  // unsupported: payload can't possibly fit
    std::memcpy(blob.data() + header.manifest_offset, &entry, sizeof(entry));

    EXPECT_FALSE(hsm::Reader::open(blob.data(), blob.size()).has_value());
}

TEST(HsmReaderTest, rejects_buffer_smaller_than_container_size) {
    const auto blob = make_sample_reader_container();
    // One byte short of what Header::container_size actually claims for this blob.
    EXPECT_FALSE(hsm::Reader::open(blob.data(), blob.size() - 1).has_value());
}

TEST(HsmReaderTest, rejects_mismatched_major_version) {
    auto blob = make_sample_reader_container();
    auto header = hsm::Header::view(blob.data());
    header.version_major = hsm::FormatVersion::major + 1;
    std::memcpy(blob.data(), &header, sizeof(header));

    EXPECT_FALSE(hsm::Reader::open(blob.data(), blob.size()).has_value());
}

TEST(HsmReaderTest, accepts_container_with_different_minor_version) {
    // Only version_major is part of the compatibility contract - a minor-version bump must stay readable.
    auto blob = make_sample_reader_container();
    auto header = hsm::Header::view(blob.data());
    header.version_minor = hsm::FormatVersion::minor + 1;
    std::memcpy(blob.data(), &header, sizeof(header));

    EXPECT_TRUE(hsm::Reader::open(blob.data(), blob.size()).has_value());
}

// --- Reader over a std::istream& source: same class, same output type, just a different open() -------

TEST(HsmReaderStreamTest, open_rejects_invalid_stream) {
    auto blob = make_container({});
    blob[0] = 'X';  // corrupt magic
    auto stream = make_stream(blob);
    EXPECT_FALSE(hsm::Reader::open(stream).has_value());
}

TEST(HsmReaderStreamTest, open_rejects_a_stream_already_in_a_failed_state) {
    std::stringstream stream;
    stream.setstate(std::ios::failbit);
    EXPECT_FALSE(hsm::Reader::open(stream).has_value());
}

TEST(HsmReaderStreamTest, open_accepts_valid_container_at_nonzero_stream_position) {
    const auto blob = make_sample_reader_container();
    auto stream = make_stream(blob, /*prefix_size=*/16);
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader.has_value());
}

TEST(HsmReaderStreamTest, open_accepts_shared_context_magic_and_reads_its_sections) {
    const auto blob = make_container(
        {
            {make_pointer_entry(fake_device_id, hsm::SectionTag::make_device_tag(/*local_id=*/1, /*is_inline=*/false)),
             "shared-context-payload"},
        },
        hsm::BlobMagic::multi);
    auto stream = make_stream(blob);
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader.has_value());

    const auto shard = reader->section(fake_device_id, hsm::SectionTag::make_device_tag(1, false).id());
    ASSERT_TRUE(shard.has_value());
    EXPECT_EQ(to_string(*shard), "shared-context-payload");
}

TEST(HsmReaderStreamTest, model_id_and_model_sections_read_correctly) {
    const auto blob = make_sample_reader_container();
    auto stream = make_stream(blob);
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader.has_value());

    const auto model_id = reader->section(hsm::model_id);
    ASSERT_TRUE(model_id.has_value());
    ASSERT_EQ(model_id->size(), 24u);
    const auto model_id_view = model_id->view();  // inline - always viewable, even without a buffer
    ASSERT_TRUE(model_id_view.has_value());
    EXPECT_EQ(static_cast<uint8_t>(model_id_view->data()[0]), 0xAA);

    const auto model = reader->section(hsm::model);
    ASSERT_TRUE(model.has_value());
    EXPECT_EQ(to_string(*model), "compiled-model-bytes");
}

TEST(HsmReaderStreamTest, section_read_reads_only_the_requested_slice_without_a_buffer) {
    const std::string payload = "compiled-model-bytes";
    const auto blob = make_sample_reader_container();
    auto stream = make_stream(blob);
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader.has_value());

    // view() has no zero-copy option here (no addressable buffer) - read() is the only way to get part of
    // a section without also reading (or allocating for) the rest.
    const auto section = reader->section(hsm::model);
    ASSERT_TRUE(section.has_value());
    EXPECT_FALSE(section->view().has_value());

    std::vector<std::byte> destination(5);
    ASSERT_TRUE(section->read(9, 5, destination.data()));
    EXPECT_EQ(to_string(destination), payload.substr(9, 5));
}

TEST(HsmReaderStreamTest, section_read_supports_repeated_calls_without_a_buffer) {
    const std::string payload = "compiled-model-bytes";
    const auto blob = make_sample_reader_container();
    auto stream = make_stream(blob);
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader.has_value());

    const auto section = reader->section(hsm::model);
    ASSERT_TRUE(section.has_value());

    // Read the first 9 bytes, decide (at "runtime") what to read next, then read the rest - without ever
    // materializing the whole section up front.
    std::vector<std::byte> head(9);
    ASSERT_TRUE(section->read(0, 9, head.data()));
    EXPECT_EQ(to_string(head), payload.substr(0, 9));

    std::vector<std::byte> rest(payload.size() - 9);
    ASSERT_TRUE(section->read(9, rest.size(), rest.data()));
    EXPECT_EQ(to_string(rest), payload.substr(9));
}

TEST(HsmReaderStreamTest, section_outlives_the_reader_it_came_from) {
    const std::string payload = "compiled-model-bytes";
    const auto blob = make_sample_reader_container();
    auto stream = make_stream(blob);

    std::optional<hsm::Section> section;
    {
        auto reader = hsm::Reader::open(stream);
        ASSERT_TRUE(reader.has_value());
        section = reader->section(hsm::model);
        ASSERT_TRUE(section.has_value());
    }  // `reader` is destroyed here - `section` must not depend on it, only on `stream`.

    std::vector<std::byte> destination(5);
    ASSERT_TRUE(section->read(9, 5, destination.data()));
    EXPECT_EQ(to_string(destination), payload.substr(9, 5));
}

TEST(HsmReaderStreamTest, decode_falls_back_to_read_when_no_view_is_available) {
    const auto blob = make_sample_reader_container();
    auto stream = make_stream(blob);
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader.has_value());

    struct ModelData {
        std::string bytes;
    };
    // Exercises make_section_decoder()'s fallback path: no addressable buffer here, so it must
    // resolve via Section::to_bytes() instead of view() - same parse callback either way.
    const auto decode_model = hsm::make_section_decoder<ModelData>([](ov::util::MemoryView view) {
        return std::make_optional(ModelData{std::string(reinterpret_cast<const char*>(view.data()), view.size())});
    });

    const auto decoded = reader->decode(hsm::model, decode_model);
    ASSERT_TRUE(decoded.has_value());
    EXPECT_EQ(decoded->bytes, "compiled-model-bytes");
}

TEST(HsmReaderStreamTest, move_constructed_reader_still_reads_sections) {
    const auto blob = make_sample_reader_container();
    auto stream = make_stream(blob);
    auto original = hsm::Reader::open(stream);
    ASSERT_TRUE(original.has_value());

    const hsm::Reader moved(std::move(*original));

    const auto section = moved.section(hsm::model);
    ASSERT_TRUE(section.has_value());
    EXPECT_EQ(to_string(*section), "compiled-model-bytes");
}

TEST(HsmReaderStreamTest, move_assignment_replaces_existing_reader) {
    const auto blob_a = make_sample_reader_container();
    auto stream_a = make_stream(blob_a);
    auto reader_a = hsm::Reader::open(stream_a);
    ASSERT_TRUE(reader_a.has_value());

    const auto blob_b = make_container({
        {make_inline_entry(hsm::any_device_id, hsm::model_id_tag, {0xBB}), {}},
    });
    auto stream_b = make_stream(blob_b);
    auto reader_b = hsm::Reader::open(stream_b);
    ASSERT_TRUE(reader_b.has_value());

    *reader_a = std::move(*reader_b);

    const auto section = reader_a->section(hsm::model_id);
    ASSERT_TRUE(section.has_value());
    const auto view = section->view();
    ASSERT_TRUE(view.has_value());
    EXPECT_EQ(static_cast<uint8_t>(view->data()[0]), 0xBB);
}

TEST(HsmReaderStreamTest, view_siblings_work_for_inline_but_not_pointer_mode_entries) {
    const auto blob = make_sample_reader_container();
    auto stream = make_stream(blob);
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader.has_value());

    const auto model_id = reader->section(hsm::model_id);
    ASSERT_TRUE(model_id.has_value());
    const auto model_id_view = model_id->view();
    ASSERT_TRUE(model_id_view.has_value());
    ASSERT_EQ(model_id_view->size(), 24u);
    EXPECT_EQ(static_cast<uint8_t>(model_id_view->data()[0]), 0xAA);

    const auto model = reader->section(hsm::model);
    ASSERT_TRUE(model.has_value());
    EXPECT_FALSE(model->view().has_value());
}

TEST(HsmReaderStreamTest, common_sections_absent_when_manifest_is_empty) {
    const auto blob = make_container({});
    auto stream = make_stream(blob);
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader.has_value());

    EXPECT_FALSE(reader->section(hsm::model_id).has_value());
    EXPECT_FALSE(reader->section(hsm::model).has_value());
    EXPECT_FALSE(reader->section(hsm::runtime_requirements).has_value());
}

TEST(HsmReaderStreamTest, sections_returns_every_matching_entry) {
    const auto shard_tag = hsm::SectionTag::make_device_tag(/*local_id=*/2, /*is_inline=*/false);
    const auto blob = make_container({
        {make_pointer_entry(fake_device_id, shard_tag), "shard-0"},
        {make_pointer_entry(fake_device_id, shard_tag), "shard-1"},
    });
    auto stream = make_stream(blob);
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader.has_value());

    const auto sections = reader->sections(fake_device_id, shard_tag.id());
    ASSERT_EQ(sections.size(), 2u);
    EXPECT_EQ(to_string(sections[0]), "shard-0");
    EXPECT_EQ(to_string(sections[1]), "shard-1");
}

TEST(HsmReaderStreamTest, read_sections_dispatches_same_handler_type_as_memory_backed_reader) {
    const auto blob = make_sample_reader_container();
    auto stream = make_stream(blob);
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader.has_value());

    RecordingHandler extension(fake_device_id, /*tag_id=*/hsm::core_tag_id_range_end + 1);
    EXPECT_EQ(reader->read_sections({&extension}), 1u);
    EXPECT_EQ(extension.last_payload, "device-specific-payload");
}

TEST(HsmReaderStreamTest, read_sections_skips_an_entry_whose_bounds_validate_but_whose_read_fails) {
    const auto blob = make_container({
        {make_pointer_entry(fake_device_id, hsm::SectionTag::make_device_tag(1, false)), "unreadable-payload"},
    });
    ::testing::NiceMock<MockStreamBuf> buf(blob);
    {
        ::testing::InSequence seq;
        EXPECT_CALL(buf, xsgetn).Times(2);                        // open()'s header + manifest reads
        EXPECT_CALL(buf, xsgetn).WillOnce(::testing::Return(0));  // the one payload read this test targets
        EXPECT_CALL(buf, xsgetn).Times(::testing::AnyNumber());
    }
    std::istream stream(&buf);
    const auto reader = hsm::Reader::open(stream);
    ASSERT_TRUE(reader.has_value());

    RecordingHandler handler(fake_device_id,
                             hsm::SectionTag::make_device_tag(/*local_id=*/1, /*is_inline=*/false).id());
    EXPECT_EQ(reader->read_sections({&handler}), 0u);
    EXPECT_EQ(handler.handled_count, 0u);
}

TEST(HsmReaderStreamTest, rejects_invalid_manifest_offset) {
    auto blob = make_sample_reader_container();
    auto header = hsm::Header::view(blob.data());
    header.manifest_offset = sizeof(hsm::Header) - 1;  // would start inside the header itself
    header.container_size = blob.size();               // ensure container_size is consistent with the actual blob size
    std::memcpy(blob.data(), &header, sizeof(header));

    auto stream = make_stream(blob);
    EXPECT_FALSE(hsm::Reader::open(stream).has_value());
}

TEST(HsmReaderStreamTest, rejects_container_size_exceeding_available_stream_bytes) {
    auto blob = make_sample_reader_container();
    auto header = hsm::Header::view(blob.data());
    header.container_size += 100;  // claims more bytes than the stream actually holds
    std::memcpy(blob.data(), &header, sizeof(header));

    auto stream = make_stream(blob);  // not padded to match the inflated container_size
    EXPECT_FALSE(hsm::Reader::open(stream).has_value());
}

TEST(HsmReaderStreamTest, rejects_invalid_section_offset) {
    auto blob = make_sample_reader_container();
    const auto header = hsm::Header::view(blob.data());

    // Second manifest entry is the pointer-mode "model" tag; corrupt its offset out of bounds.
    const auto model_entry_offset = header.manifest_offset + sizeof(hsm::ManifestEntry);
    hsm::ManifestEntry entry{};
    std::memcpy(&entry, blob.data() + model_entry_offset, sizeof(entry));
    entry.offset = blob.size() + 1;
    std::memcpy(blob.data() + model_entry_offset, &entry, sizeof(entry));

    auto stream = make_stream(blob);
    EXPECT_FALSE(hsm::Reader::open(stream).has_value());
}

TEST(HsmReaderStreamTest, rejects_mismatched_major_version) {
    auto blob = make_sample_reader_container();
    auto header = hsm::Header::view(blob.data());
    header.version_major = hsm::FormatVersion::major + 1;
    std::memcpy(blob.data(), &header, sizeof(header));

    auto stream = make_stream(blob);
    EXPECT_FALSE(hsm::Reader::open(stream).has_value());
}

TEST(HsmReaderStreamTest, accepts_container_with_different_minor_version) {
    auto blob = make_sample_reader_container();
    auto header = hsm::Header::view(blob.data());
    header.version_minor = hsm::FormatVersion::minor + 1;
    std::memcpy(blob.data(), &header, sizeof(header));

    auto stream = make_stream(blob);
    EXPECT_TRUE(hsm::Reader::open(stream).has_value());
}

}  // namespace ov::test
