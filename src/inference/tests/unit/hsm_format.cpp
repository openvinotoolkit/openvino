// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_format.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace ov::test {
namespace {

// Fixed size of a single manifest entry (see @ref ov::runtime::HSMHeader::manifest_size). ManifestEntry itself
// isn't defined in hsm_format.hpp right now, so tests use this plain byte count instead of sizeof(...).
constexpr size_t k_manifest_entry_size = 32;

// Builds the raw bytes of a structurally-valid HSM container: HSMHeader, then `section_payload` (if any),
// then `manifest` (raw, pre-built manifest table bytes). Only lays out the container's top-level regions
// (header / section payload / manifest); doesn't know or care what any given section payload or manifest
// entry means - callers pass already-serialized bytes, not typed ManifestEntry objects.
std::vector<uint8_t> make_container(const runtime::BlobMagic& magic,
                                    const std::vector<uint8_t>& section_payload,
                                    const std::vector<uint8_t>& manifest) {
    runtime::HSMHeader header{};
    header.magic = magic;
    header.version_major = runtime::HSMFormatVersion::major;
    header.version_minor = runtime::HSMFormatVersion::minor;
    header.manifest_offset = sizeof(runtime::HSMHeader) + section_payload.size();
    header.manifest_size = manifest.size();
    header.total_size = header.manifest_offset + header.manifest_size;

    std::vector<uint8_t> buffer(header.total_size);
    std::memcpy(buffer.data(), &header, sizeof(header));
    if (!section_payload.empty()) {
        std::memcpy(buffer.data() + sizeof(header), section_payload.data(), section_payload.size());
    }
    if (!manifest.empty()) {
        std::memcpy(buffer.data() + header.manifest_offset, manifest.data(), manifest.size());
    }
    return buffer;
}

// Single compiled model container.
std::vector<uint8_t> make_single_blob_container(const std::vector<uint8_t>& section_payload,
                                                const std::vector<uint8_t>& manifest) {
    return make_container(runtime::BlobMagic::single, section_payload, manifest);
}

// One container tagged as part of a multi-blob file (Story 12/13, forward-looking).
std::vector<uint8_t> make_multi_container(const std::vector<uint8_t>& section_payload,
                                          const std::vector<uint8_t>& manifest) {
    return make_container(runtime::BlobMagic::multi, section_payload, manifest);
}

// A multi-blob file: `blob_count` independent MULTI-magic containers concatenated back-to-back, each with
// its own header/manifest. The exact multi-container framing isn't finalized yet (Story 12/13), but every
// container is self-describing via `total_size`, so a reader can hop from one to the next by adding it to
// the current container's start offset - back-to-back concatenation is enough to exercise that here.
// `blob_count == 1` is a valid (degenerate) multi-blob file containing a single blob.
std::vector<uint8_t> make_multi_blob_container(size_t blob_count) {
    std::vector<uint8_t> buffer;
    for (size_t i = 0; i < blob_count; ++i) {
        const auto blob = make_multi_container({}, {});
        buffer.insert(buffer.end(), blob.begin(), blob.end());
    }
    return buffer;
}

// A multi-blob file matching HSMMultiBlobView's model: one mandatory shared-context container
// (BlobMagic::multi, empty) followed by `blob_count` actual model containers (BlobMagic::single, empty
// payload/manifest).
std::vector<uint8_t> make_multi_blob_file(size_t blob_count) {
    std::vector<uint8_t> buffer = make_multi_container({}, {});
    for (size_t i = 0; i < blob_count; ++i) {
        const auto blob = make_single_blob_container({}, {});
        buffer.insert(buffer.end(), blob.begin(), blob.end());
    }
    return buffer;
}

// Sample container shared by the HsmHeaderTest.* tests below: a header describing 2 manifest entries and a
// 2-byte "OV" section payload. Manifest content is dummy placeholder bytes (just the correct byte count,
// `2 * kManifestEntrySize`) - no test here reads specific entry field values.
std::vector<uint8_t> make_sample_single_blob_container() {
    const std::vector<uint8_t> manifest(2 * k_manifest_entry_size, 0xCD);
    return make_single_blob_container({'O', 'V'}, manifest);
}

// Sample container with real entries: a Core "model_id" inline entry, and a "model" pointer entry pointing
// at a 2-byte "OV" section payload - used to exercise HSMContainerView's header/manifest/section access.
std::vector<uint8_t> make_sample_container_with_entries() {
    runtime::ManifestEntry id_entry{};
    id_entry.tag = runtime::model_id_tag();
    id_entry.inline_bytes = {0xAA, 0xBB, 0xCC, 0xDD};

    runtime::ManifestEntry model_entry{};
    model_entry.tag = runtime::model_tag();
    model_entry.offset = sizeof(runtime::HSMHeader);
    model_entry.size = 2;

    std::vector<uint8_t> manifest(2 * sizeof(runtime::ManifestEntry));
    std::memcpy(manifest.data(), &id_entry, sizeof(id_entry));
    std::memcpy(manifest.data() + sizeof(id_entry), &model_entry, sizeof(model_entry));

    return make_single_blob_container({'O', 'V'}, manifest);
}

// Minimal IHsmSectionExtension: recognizes one (device, tag id) pair, records the section size it saw.
class RecordingExtension : public runtime::IHsmSectionExtension {
public:
    static constexpr runtime::DeviceId owned_device = 7;
    static constexpr uint32_t owned_tag_id = 42;

    bool read_section(const runtime::ManifestEntry& entry, ov::util::MemoryView section) override {
        if (entry.device != owned_device || entry.tag.id() != owned_tag_id) {
            return false;
        }
        last_section_size = section.size();
        return true;
    }

    size_t last_section_size = 0;
};

}  // namespace

// --- HSM wire-format layout/version compatibility --------------------------------------------------------
// Pinned to HSMFormatVersion::major == 1. A failure in this fixture means the on-disk byte layout changed:
static_assert(runtime::HSMFormatVersion::major == 1,
              "HSMFormatVersion::major changed - review HsmFormatLayoutCompatibilityTest, update the pinned "
              "layout checks below to match the new format, then update this assert.");

class HsmFormatLayoutCompatibilityTest : public ::testing::Test {};

TEST_F(HsmFormatLayoutCompatibilityTest, hsm_header_layout) {
    static_assert(sizeof(runtime::HSMMagicType) == 5, "HSMMagicType width changed");
    static_assert(sizeof(runtime::HSMHeader) == 32, "HSMHeader total size changed");
    static_assert(offsetof(runtime::HSMHeader, magic) == 0, "HSMHeader::magic offset changed");
    static_assert(offsetof(runtime::HSMHeader, version_major) == 5, "HSMHeader::version_major offset changed");
    static_assert(offsetof(runtime::HSMHeader, version_minor) == 7, "HSMHeader::version_minor offset changed");
    static_assert(offsetof(runtime::HSMHeader, total_size) == 8, "HSMHeader::total_size offset changed");
    static_assert(offsetof(runtime::HSMHeader, manifest_offset) == 16, "HSMHeader::manifest_offset offset changed");
    static_assert(offsetof(runtime::HSMHeader, manifest_size) == 24, "HSMHeader::manifest_size offset changed");

    // Runtime echo, for visibility in test reports (the static_asserts above already block the build on a break).
    EXPECT_EQ(sizeof(runtime::HSMHeader), 32u);
    EXPECT_EQ(offsetof(runtime::HSMHeader, manifest_size), 24u);
}

TEST_F(HsmFormatLayoutCompatibilityTest, manifest_entry_layout) {
    static_assert(sizeof(runtime::ManifestEntry) == 32, "ManifestEntry total size changed");
    static_assert(offsetof(runtime::ManifestEntry, device) == 0, "ManifestEntry::device offset changed");
    static_assert(offsetof(runtime::ManifestEntry, tag) == 1, "ManifestEntry::tag offset changed");
    static_assert(sizeof(runtime::SectionTag) == 3, "SectionTag width changed");
    static_assert(offsetof(runtime::ManifestEntry, tag_reserved) == 4, "ManifestEntry::tag_reserved offset changed");
    static_assert(offsetof(runtime::ManifestEntry, offset) == 8, "ManifestEntry::offset offset changed");
    static_assert(offsetof(runtime::ManifestEntry, size) == 16, "ManifestEntry::size offset changed");
    static_assert(offsetof(runtime::ManifestEntry, pointer_reserved) == 24,
                  "ManifestEntry::pointer_reserved offset changed");
    static_assert(offsetof(runtime::ManifestEntry, inline_bytes) == 8, "ManifestEntry::inline_bytes offset changed");
    static_assert(sizeof(runtime::ManifestEntry{}.inline_bytes) == 24, "ManifestEntry::inline_bytes width changed");

    EXPECT_EQ(sizeof(runtime::ManifestEntry), 32u);
}

TEST_F(HsmFormatLayoutCompatibilityTest, blob_magic_is_compile_time_comparable) {
    // BlobMagic::operator==/!= must be usable in a constant expression, not just nominally marked constexpr.
    static_assert(runtime::BlobMagic::single == runtime::BlobMagic::single,
                  "BlobMagic::operator== must be constexpr-usable");
    static_assert(runtime::BlobMagic::single != runtime::BlobMagic::multi,
                  "BlobMagic::operator!= must be constexpr-usable");
    SUCCEED();
}

TEST_F(HsmFormatLayoutCompatibilityTest, section_tag_packs_id_and_mode_at_compile_time) {
    static_assert(runtime::SectionTag::make(1, true).id() == 1u, "tag id round-trip (inline)");
    static_assert(runtime::SectionTag::make(1, false).id() == 1u, "tag id round-trip (pointer)");
    static_assert(runtime::SectionTag::make(0x7FFFFF, true).id() == 0x7FFFFFu, "tag id round-trip (max 23-bit id)");

    static_assert(runtime::SectionTag::make(1, true).is_inline(), "make(id, true) must produce an inline-mode tag");
    static_assert(!runtime::SectionTag::make(1, true).is_pointer(),
                  "inline-mode tag must not also read as pointer-mode");
    static_assert(runtime::SectionTag::make(2, false).is_pointer(), "make(id, false) must produce a pointer-mode tag");
    static_assert(!runtime::SectionTag::make(2, false).is_inline(),
                  "pointer-mode tag must not also read as inline-mode");

    SUCCEED();
}

TEST_F(HsmFormatLayoutCompatibilityTest, core_tags_have_fixed_mode) {
    static_assert(runtime::model_id_tag().id() == runtime::model_id, "model_id_tag id");
    static_assert(runtime::model_id_tag().is_inline(), "model_id_tag must always be inline-mode");

    static_assert(runtime::model_tag().id() == runtime::model, "model_tag id");
    static_assert(runtime::model_tag().is_pointer(), "model_tag must always be pointer-mode");

    static_assert(runtime::runtime_requirements_tag().id() == runtime::runtime_requirements,
                  "runtime_requirements_tag id");
    static_assert(runtime::runtime_requirements_tag().is_pointer(),
                  "runtime_requirements_tag must always be pointer-mode");

    SUCCEED();
}

TEST_F(HsmFormatLayoutCompatibilityTest, device_tags_cannot_collide_with_core_tags) {
    // A device tag built from local id 0 must never land in the Core-owned range, no matter how small the
    // local id is - this is the whole point of make_device_tag() vs. picking a raw absolute id by hand.
    static_assert(runtime::make_device_tag(0, true).id() == runtime::core_tag_id_range_end,
                  "make_device_tag(0, ...) must land exactly at the range boundary");
    static_assert(runtime::make_device_tag(0, true).id() >= runtime::core_tag_id_range_end,
                  "device tag ids must never fall below core_tag_id_range_end");
    static_assert(runtime::model_id < runtime::core_tag_id_range_end, "model_id must stay below core_tag_id_range_end");
    static_assert(runtime::model < runtime::core_tag_id_range_end, "model must stay below core_tag_id_range_end");

    SUCCEED();
}
// --- HSM wire-format layout/version compatibility end ----------------------------------------------------

TEST(HsmHeaderTest, check_single_blob_header) {
    const auto blob = make_sample_single_blob_container();
    ASSERT_GE(blob.size(), sizeof(runtime::HSMHeader));

    const auto header = runtime::HSMHeader::view(blob.data());
    EXPECT_EQ(header.magic, runtime::BlobMagic::single);
    EXPECT_EQ(header.version_major, runtime::HSMFormatVersion::major);
    EXPECT_EQ(header.version_minor, runtime::HSMFormatVersion::minor);
    EXPECT_EQ(header.total_size, blob.size());
    EXPECT_EQ(header.manifest_offset, sizeof(runtime::HSMHeader) + 2u);  // 2 bytes of section payload
    EXPECT_EQ(header.manifest_size, 2 * k_manifest_entry_size);          // 2 entries in the manifest
};

TEST(HsmHeaderTest, check_multi_blob_single_blob) {
    const auto blob = make_multi_blob_container(1);
    ASSERT_GE(blob.size(), sizeof(runtime::HSMHeader));

    const auto header = runtime::HSMHeader::view(blob.data());
    EXPECT_EQ(header.magic, runtime::BlobMagic::multi);
    EXPECT_EQ(header.version_major, runtime::HSMFormatVersion::major);
    EXPECT_EQ(header.version_minor, runtime::HSMFormatVersion::minor);
    EXPECT_EQ(header.total_size, blob.size());
    EXPECT_EQ(header.manifest_offset, sizeof(runtime::HSMHeader));
    EXPECT_EQ(header.manifest_size, 0u);  // no entries in the manifest
};

TEST(HsmHeaderTest, check_multi_blob_two_blobs) {
    const auto blob = make_multi_blob_container(2);
    ASSERT_GE(blob.size(), 2 * sizeof(runtime::HSMHeader));

    const auto header1 = runtime::HSMHeader::view(blob.data());
    EXPECT_EQ(header1.magic, runtime::BlobMagic::multi);
    EXPECT_EQ(header1.version_major, runtime::HSMFormatVersion::major);
    EXPECT_EQ(header1.version_minor, runtime::HSMFormatVersion::minor);
    EXPECT_EQ(header1.total_size, sizeof(runtime::HSMHeader));
    EXPECT_EQ(header1.manifest_offset, sizeof(runtime::HSMHeader));
    EXPECT_EQ(header1.manifest_size, 0u);  // no entries in the manifest

    const auto header2 = runtime::HSMHeader::view(blob.data() + header1.total_size);
    EXPECT_EQ(header2.magic, runtime::BlobMagic::multi);
    EXPECT_EQ(header2.version_major, runtime::HSMFormatVersion::major);
    EXPECT_EQ(header2.version_minor, runtime::HSMFormatVersion::minor);
    EXPECT_EQ(header2.total_size, sizeof(runtime::HSMHeader));
    EXPECT_EQ(header2.manifest_offset, sizeof(runtime::HSMHeader));
    EXPECT_EQ(header2.manifest_size, 0u);  // no entries in the manifest
}

TEST(HsmContainerViewTest, reads_header) {
    const auto blob = make_sample_container_with_entries();
    const runtime::HSMContainerView view(blob.data(), blob.size());
    EXPECT_EQ(view.size(), blob.size());
    EXPECT_EQ(view.header().magic, runtime::BlobMagic::single);
}

TEST(HsmContainerViewTest, reads_manifest_entries) {
    const auto blob = make_sample_container_with_entries();
    const runtime::HSMContainerView view(blob.data(), blob.size());

    ASSERT_EQ(view.manifest_count(), 2u);
    const auto* manifest = &view.manifest();

    const auto& id_entry = manifest[0];
    EXPECT_EQ(id_entry.tag.id(), runtime::model_id);
    EXPECT_TRUE(id_entry.tag.is_inline());
    EXPECT_EQ(id_entry.inline_bytes[0], 0xAA);

    const auto& model_entry = manifest[1];
    EXPECT_EQ(model_entry.tag.id(), runtime::model);
    EXPECT_TRUE(model_entry.tag.is_pointer());
    EXPECT_EQ(model_entry.size, 2u);
}

TEST(HsmContainerViewTest, reads_pointer_section) {
    const auto blob = make_sample_container_with_entries();
    const runtime::HSMContainerView view(blob.data(), blob.size());
    const auto& model_entry = (&view.manifest())[1];
    const auto section = view.section(model_entry);
    ASSERT_EQ(section.size(), 2u);
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(section.data()), section.size()), "OV");
}

TEST(HsmContainerViewTest, section_rejects_inline_mode_entry) {
    const auto blob = make_sample_container_with_entries();
    const runtime::HSMContainerView view(blob.data(), blob.size());

    runtime::ManifestEntry entry{};
    entry.tag = runtime::model_id_tag();  // inline-mode: offset/size below don't refer to a real section
    entry.offset = sizeof(runtime::HSMHeader);
    entry.size = 2;

    EXPECT_EQ(view.section(entry).size(), 0u);
}

TEST(HsmContainerViewTest, section_rejects_out_of_bounds_offset) {
    const auto blob = make_sample_container_with_entries();
    const runtime::HSMContainerView view(blob.data(), blob.size());

    runtime::ManifestEntry entry{};
    entry.tag = runtime::model_tag();
    entry.offset = view.size() + 1;
    entry.size = 1;

    EXPECT_EQ(view.section(entry).size(), 0u);
}

TEST(HsmContainerViewTest, section_rejects_out_of_bounds_size) {
    const auto blob = make_sample_container_with_entries();
    const runtime::HSMContainerView view(blob.data(), blob.size());

    runtime::ManifestEntry entry{};
    entry.tag = runtime::model_tag();
    entry.offset = 0;
    entry.size = view.size() + 1;  // fits at offset 0 alone, but overruns the buffer

    EXPECT_EQ(view.section(entry).size(), 0u);
}

TEST(HsmContainerViewValidateTest, accepts_well_formed_container) {
    const auto blob = make_sample_container_with_entries();
    const runtime::HSMContainerView view(blob.data(), blob.size());
    EXPECT_TRUE(view.validate());
}

TEST(HsmContainerViewValidateTest, rejects_bad_magic) {
    auto blob = make_sample_container_with_entries();
    blob[0] = 'X';  // corrupt HSMHeader::magic
    const runtime::HSMContainerView view(blob.data(), blob.size());
    EXPECT_FALSE(view.validate());
}

TEST(HsmContainerViewValidateTest, rejects_mismatched_major_version) {
    auto blob = make_sample_container_with_entries();
    auto header = runtime::HSMHeader::view(blob.data());
    header.version_major = runtime::HSMFormatVersion::major + 1;
    std::memcpy(blob.data(), &header, sizeof(header));

    const runtime::HSMContainerView view(blob.data(), blob.size());
    EXPECT_FALSE(view.validate());
}

TEST(HsmContainerViewValidateTest, rejects_buffer_smaller_than_total_size) {
    const auto blob = make_sample_container_with_entries();
    // View sees fewer bytes than HSMHeader::total_size claims.
    const runtime::HSMContainerView view(blob.data(), blob.size() - 1);
    EXPECT_FALSE(view.validate());
}

TEST(HsmContainerViewValidateTest, rejects_out_of_bounds_manifest_offset) {
    auto blob = make_sample_container_with_entries();
    auto header = runtime::HSMHeader::view(blob.data());
    header.manifest_offset = std::numeric_limits<runtime::HSMOffsetType>::max();
    std::memcpy(blob.data(), &header, sizeof(header));

    const runtime::HSMContainerView view(blob.data(), blob.size());
    EXPECT_FALSE(view.validate());
}

TEST(HsmContainerViewValidateTest, rejects_manifest_offset_inside_header) {
    auto blob = make_sample_container_with_entries();
    auto header = runtime::HSMHeader::view(blob.data());
    header.manifest_offset = sizeof(runtime::HSMHeader) - 1;  // would start reading manifest inside the header
    std::memcpy(blob.data(), &header, sizeof(header));

    const runtime::HSMContainerView view(blob.data(), blob.size());
    EXPECT_FALSE(view.validate());
}

TEST(HsmContainerViewValidateTest, accepts_container_with_empty_manifest) {
    const auto blob = make_single_blob_container({}, {});  // no section payload, no manifest entries
    const runtime::HSMContainerView view(blob.data(), blob.size());
    ASSERT_EQ(view.manifest_count(), 0u);
    EXPECT_TRUE(view.validate());
}

TEST(HsmContainerViewValidateTest, rejects_out_of_bounds_section_offset) {
    auto blob = make_sample_container_with_entries();
    const auto header = runtime::HSMHeader::view(blob.data());

    // Second manifest entry is the pointer-mode "model" tag.
    const auto entry_offset = header.manifest_offset + sizeof(runtime::ManifestEntry);
    runtime::ManifestEntry entry{};
    std::memcpy(&entry, blob.data() + entry_offset, sizeof(entry));
    entry.offset = std::numeric_limits<runtime::HSMOffsetType>::max();
    std::memcpy(blob.data() + entry_offset, &entry, sizeof(entry));

    const runtime::HSMContainerView view(blob.data(), blob.size());
    EXPECT_FALSE(view.validate());
}

TEST(HsmMultiBlobViewTest, empty_buffer_has_no_blobs) {
    const runtime::HSMMultiBlobView view(static_cast<const uint8_t*>(nullptr), 0);
    EXPECT_EQ(view.blob_count(), 0u);
}

TEST(HsmMultiBlobViewTest, reads_single_blob) {
    const auto blob = make_multi_blob_file(1);
    const runtime::HSMMultiBlobView view(blob.data(), blob.size());

    ASSERT_EQ(view.blob_count(), 1u);
    EXPECT_EQ(view.blob_at(0).header().magic, runtime::BlobMagic::single);
}

TEST(HsmMultiBlobViewTest, reads_multiple_blobs) {
    const auto blob = make_multi_blob_file(3);
    const runtime::HSMMultiBlobView view(blob.data(), blob.size());

    ASSERT_EQ(view.blob_count(), 3u);
    for (size_t i = 0; i < view.blob_count(); ++i) {
        EXPECT_EQ(view.blob_at(i).header().magic, runtime::BlobMagic::single);
    }
}

TEST(HsmMultiBlobViewTest, skips_optional_shared_context_between_blobs) {
    auto buffer = make_multi_blob_file(1);                    // mandatory shared context + 1 blob
    const auto extra_context = make_multi_container({}, {});  // optional shared-context update
    buffer.insert(buffer.end(), extra_context.begin(), extra_context.end());
    const auto blob1 = make_single_blob_container({}, {});
    buffer.insert(buffer.end(), blob1.begin(), blob1.end());

    const runtime::HSMMultiBlobView view(buffer.data(), buffer.size());
    ASSERT_EQ(view.blob_count(), 2u);  // the extra shared-context container doesn't count as a blob
    EXPECT_EQ(view.blob_at(0).header().magic, runtime::BlobMagic::single);
    EXPECT_TRUE(view.blob_at(0).validate());
    EXPECT_EQ(view.blob_at(1).header().magic, runtime::BlobMagic::single);
}

TEST(HsmMultiBlobViewTest, stops_on_oversized_total_size) {
    auto blob = make_multi_blob_file(1);
    auto header = runtime::HSMHeader::view(blob.data());  // corrupt the mandatory shared context's header
    header.total_size = std::numeric_limits<runtime::HSMSizeType>::max();
    std::memcpy(blob.data(), &header, sizeof(header));

    const runtime::HSMMultiBlobView view(blob.data(), blob.size());
    EXPECT_EQ(view.blob_count(), 0u);       // can't even get past the corrupt first container
    EXPECT_EQ(view.blob_at(0).size(), 0u);  // out-of-range -> empty view, not a crash
}

TEST(HsmMultiBlobViewTest, stops_on_invalid_magic) {
    auto blob = make_multi_blob_file(1);
    const auto second_container_offset = sizeof(runtime::HSMHeader);
    blob[second_container_offset] = 'X';  // corrupt the blob's magic to neither single nor multi

    const runtime::HSMMultiBlobView view(blob.data(), blob.size());
    EXPECT_EQ(view.blob_count(), 0u);  // shared context is skipped fine, but the blob itself is unreadable
}

TEST(HsmMultiBlobViewTest, blob_view_excludes_following_containers) {
    auto buffer = make_multi_container({}, {});  // mandatory shared context
    const auto blob0 = make_single_blob_container({}, {});
    const auto blob1 = make_single_blob_container({'O', 'V'}, {});  // different size than blob0
    buffer.insert(buffer.end(), blob0.begin(), blob0.end());
    buffer.insert(buffer.end(), blob1.begin(), blob1.end());

    const runtime::HSMMultiBlobView view(buffer.data(), buffer.size());
    ASSERT_EQ(view.blob_count(), 2u);

    const auto view0 = view.blob_at(0);
    EXPECT_EQ(view0.size(), blob0.size());  // must not leak into blob1's bytes
    EXPECT_TRUE(view0.validate());
}

TEST(IHsmSectionExtensionTest, recognizes_own_device_and_tag) {
    runtime::ManifestEntry entry{};
    entry.device = RecordingExtension::owned_device;
    entry.tag = runtime::SectionTag::make(RecordingExtension::owned_tag_id, /*is_inline=*/false);

    const std::byte payload[4] = {};
    RecordingExtension extension;
    EXPECT_TRUE(extension.read_section(entry, ov::util::MemoryView{payload, 4}));
    EXPECT_EQ(extension.last_section_size, 4u);
}

TEST(IHsmSectionExtensionTest, skips_entry_it_does_not_own) {
    runtime::ManifestEntry entry{};
    entry.device = RecordingExtension::owned_device + 1;  // different device
    entry.tag = runtime::SectionTag::make(RecordingExtension::owned_tag_id, /*is_inline=*/false);

    RecordingExtension extension;
    EXPECT_FALSE(extension.read_section(entry, ov::util::MemoryView{}));
    EXPECT_EQ(extension.last_section_size, 0u);  // never called
}

// OPENVINO_DEBUG_ASSERT compiles out entirely under NDEBUG (Release builds), so this only runs in debug builds.
#ifndef NDEBUG
TEST(MakeDeviceTagTest, debug_asserts_on_id_overflow) {
    const auto out_of_range_id = runtime::max_tag_id - runtime::core_tag_id_range_end + 1;
    EXPECT_DEATH(runtime::make_device_tag(out_of_range_id, false), "");
}
#endif

}  // namespace ov::test
