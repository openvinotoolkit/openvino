// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_format.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace ov::test {
namespace hsm = ov::runtime::hsm;
namespace {

// Byte size of one manifest entry; used for raw placeholder manifest bytes below.
constexpr size_t k_manifest_entry_size = 32;

// Raw bytes of an HSM container: header, then section_payload, then manifest (already-serialized).
std::vector<uint8_t> make_container(const hsm::BlobMagic& magic,
                                    const std::vector<uint8_t>& section_payload,
                                    const std::vector<uint8_t>& manifest) {
    hsm::Header header{};
    header.magic = magic;
    header.version_major = hsm::FormatVersion::major;
    header.version_minor = hsm::FormatVersion::minor;
    header.manifest_offset = sizeof(hsm::Header) + section_payload.size();
    header.manifest_size = manifest.size();
    header.container_size = header.manifest_offset + header.manifest_size;

    std::vector<uint8_t> buffer;
    buffer.reserve(header.container_size);
    const auto* header_bytes = reinterpret_cast<const uint8_t*>(&header);
    buffer.insert(buffer.end(), header_bytes, header_bytes + sizeof(header));
    buffer.insert(buffer.end(), section_payload.begin(), section_payload.end());
    buffer.insert(buffer.end(), manifest.begin(), manifest.end());
    return buffer;
}

// Single compiled model container.
std::vector<uint8_t> make_single_blob_container(const std::vector<uint8_t>& section_payload,
                                                const std::vector<uint8_t>& manifest) {
    return make_container(hsm::BlobMagic::single, section_payload, manifest);
}

// One container tagged as part of a multi-blob file.
std::vector<uint8_t> make_multi_container(const std::vector<uint8_t>& section_payload,
                                          const std::vector<uint8_t>& manifest) {
    return make_container(hsm::BlobMagic::multi, section_payload, manifest);
}

// blob_count MULTI-magic containers concatenated back-to-back
std::vector<uint8_t> make_multi_blob_container(size_t blob_count) {
    std::vector<uint8_t> buffer;
    for (size_t i = 0; i < blob_count; ++i) {
        const auto blob = make_multi_container({}, {});
        buffer.insert(buffer.end(), blob.begin(), blob.end());
    }
    return buffer;
}

// Mandatory shared-context container (MULTI) followed by blob_count model containers (SINGLE).
std::vector<uint8_t> make_multi_blob_file(size_t blob_count) {
    std::vector<uint8_t> buffer = make_multi_container({}, {});
    for (size_t i = 0; i < blob_count; ++i) {
        const auto blob = make_single_blob_container({}, {});
        buffer.insert(buffer.end(), blob.begin(), blob.end());
    }
    return buffer;
}

// Sample container for HsmHeaderTest: 2-byte "OV" section payload, 2 dummy manifest entries.
std::vector<uint8_t> make_sample_single_blob_container() {
    const std::vector<uint8_t> manifest(2 * k_manifest_entry_size, 0xCD);
    return make_single_blob_container({'O', 'V'}, manifest);
}

// Sample container with real entries: inline model_id tag + pointer-mode model tag ("OV" payload).
std::vector<uint8_t> make_sample_container_with_entries() {
    hsm::ManifestEntry id_entry{};
    id_entry.tag = hsm::model_id_tag;
    id_entry.inline_bytes = {0xAA, 0xBB, 0xCC, 0xDD};

    hsm::ManifestEntry model_entry{};
    model_entry.tag = hsm::model_tag;
    model_entry.offset = sizeof(hsm::Header);
    model_entry.size = 2;

    std::vector<uint8_t> manifest(2 * sizeof(hsm::ManifestEntry));
    std::memcpy(manifest.data(), &id_entry, sizeof(id_entry));
    std::memcpy(manifest.data() + sizeof(id_entry), &model_entry, sizeof(model_entry));

    return make_single_blob_container({'O', 'V'}, manifest);
}

// Minimal ISectionExtension: recognizes one (device, tag id) pair, records the section size it saw.
class RecordingExtension : public hsm::ISectionExtension {
public:
    static constexpr hsm::DeviceId owned_device = 7;
    static constexpr uint32_t owned_tag_id = 42;

    bool read_section(const hsm::ManifestEntry& entry, ov::util::MemoryView section) override {
        if (entry.device != owned_device || entry.tag.id() != owned_tag_id) {
            return false;
        }
        last_section_size = section.size();
        return true;
    }

    size_t last_section_size = 0;
};

template <typename T, size_t N>
constexpr bool all_unique(const std::array<T, N>& values) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            if (values[i] == values[j]) {
                return false;
            }
        }
    }
    return true;
}

template <typename T, size_t N>
constexpr bool all_in_range(const std::array<T, N>& values, T low, T high) {
    for (size_t i = 0; i < N; ++i) {
        if (values[i] < low || !(values[i] < high)) {
            return false;
        }
    }
    return true;
}

// Every Core tag id. Listing them here is what makes the uniqueness and count guards below meaningful:
// adding a tag to Tags without adding it here fails the build.
constexpr std::array<uint32_t, 6> k_core_tag_ids{hsm::model_id,
                                                 hsm::model,
                                                 hsm::runtime_requirements,
                                                 hsm::model_struct,
                                                 hsm::weights,
                                                 hsm::compiled_options};

// Every assigned device id, pinned the same way as k_core_tag_ids.
constexpr std::array<hsm::DeviceId, 4> k_device_ids{hsm::any_device_id,
                                                    hsm::cpu_device_id,
                                                    hsm::gpu_device_id,
                                                    hsm::npu_device_id};

}  // namespace

// --- HSM wire-format layout/version compatibility --------------------------------------------------------
// Pinned to FormatVersion::major == 1. A failure in this fixture means the on-disk byte layout changed:
static_assert(hsm::FormatVersion::major == 1,
              "FormatVersion::major changed - review HsmFormatLayoutCompatibilityTest, update the pinned "
              "layout checks below to match the new format, then update this assert.");

class HsmFormatLayoutCompatibilityTest : public ::testing::Test {};

TEST_F(HsmFormatLayoutCompatibilityTest, hsm_header_layout) {
    static_assert(sizeof(hsm::MagicType) == 5, "MagicType width changed");
    static_assert(sizeof(hsm::Header) == 32, "Header total size changed");
    static_assert(offsetof(hsm::Header, magic) == 0, "Header::magic offset changed");
    static_assert(offsetof(hsm::Header, version_major) == 5, "Header::version_major offset changed");
    static_assert(offsetof(hsm::Header, version_minor) == 7, "Header::version_minor offset changed");
    static_assert(offsetof(hsm::Header, container_size) == 8, "Header::container_size offset changed");
    static_assert(offsetof(hsm::Header, manifest_offset) == 16, "Header::manifest_offset offset changed");
    static_assert(offsetof(hsm::Header, manifest_size) == 24, "Header::manifest_size offset changed");

    // Runtime echo, for visibility in test reports (the static_asserts above already block the build on a break).
    EXPECT_EQ(sizeof(hsm::Header), 32u);
    EXPECT_EQ(offsetof(hsm::Header, manifest_size), 24u);
}

TEST_F(HsmFormatLayoutCompatibilityTest, manifest_entry_layout) {
    static_assert(sizeof(hsm::ManifestEntry) == 32, "ManifestEntry total size changed");
    static_assert(offsetof(hsm::ManifestEntry, device) == 0, "ManifestEntry::device offset changed");
    static_assert(offsetof(hsm::ManifestEntry, tag) == 1, "ManifestEntry::tag offset changed");
    static_assert(sizeof(hsm::SectionTag) == 3, "SectionTag width changed");
    static_assert(offsetof(hsm::ManifestEntry, tag_reserved) == 4, "ManifestEntry::tag_reserved offset changed");
    static_assert(offsetof(hsm::ManifestEntry, offset) == 8, "ManifestEntry::offset offset changed");
    static_assert(offsetof(hsm::ManifestEntry, size) == 16, "ManifestEntry::size offset changed");
    static_assert(offsetof(hsm::ManifestEntry, pointer_reserved) == 24,
                  "ManifestEntry::pointer_reserved offset changed");
    static_assert(offsetof(hsm::ManifestEntry, inline_bytes) == 8, "ManifestEntry::inline_bytes offset changed");
    static_assert(sizeof(hsm::ManifestEntry{}.inline_bytes) == 24, "ManifestEntry::inline_bytes width changed");

    EXPECT_EQ(sizeof(hsm::ManifestEntry), 32u);
}

TEST_F(HsmFormatLayoutCompatibilityTest, blob_magic_is_compile_time_comparable) {
    // BlobMagic::operator==/!= must be usable in a constant expression, not just nominally marked constexpr.
    static_assert(hsm::BlobMagic::single == hsm::BlobMagic::single, "BlobMagic::operator== must be constexpr-usable");
    static_assert(hsm::BlobMagic::single != hsm::BlobMagic::multi, "BlobMagic::operator!= must be constexpr-usable");
    SUCCEED();
}

TEST_F(HsmFormatLayoutCompatibilityTest, section_tag_packs_id_and_mode_at_compile_time) {
    static_assert(hsm::SectionTag::make(1, true).id() == 1u, "tag id round-trip (inline)");
    static_assert(hsm::SectionTag::make(1, false).id() == 1u, "tag id round-trip (pointer)");
    static_assert(hsm::SectionTag::make(0x7FFFFF, true).id() == 0x7FFFFFu, "tag id round-trip (max 23-bit id)");

    static_assert(hsm::SectionTag::make(1, true).is_inline(), "make(id, true) must produce an inline-mode tag");
    static_assert(!hsm::SectionTag::make(1, true).is_pointer(), "inline-mode tag must not also read as pointer-mode");
    static_assert(hsm::SectionTag::make(2, false).is_pointer(), "make(id, false) must produce a pointer-mode tag");
    static_assert(!hsm::SectionTag::make(2, false).is_inline(), "pointer-mode tag must not also read as inline-mode");

    SUCCEED();
}

TEST_F(HsmFormatLayoutCompatibilityTest, core_tags_have_fixed_mode) {
    static_assert(hsm::model_id_tag.id() == hsm::model_id, "model_id_tag id");
    static_assert(hsm::model_id_tag.is_inline(), "model_id_tag must always be inline-mode");

    static_assert(hsm::model_tag.id() == hsm::model, "model_tag id");
    static_assert(hsm::model_tag.is_pointer(), "model_tag must always be pointer-mode");

    static_assert(hsm::runtime_requirements_tag.id() == hsm::runtime_requirements, "runtime_requirements_tag id");
    static_assert(hsm::runtime_requirements_tag.is_pointer(), "runtime_requirements_tag must always be pointer-mode");

    static_assert(hsm::model_struct_tag.id() == hsm::model_struct, "model_struct_tag id");
    static_assert(hsm::model_struct_tag.is_pointer(), "model_struct_tag must always be pointer-mode");

    static_assert(hsm::weights_tag.id() == hsm::weights, "weights_tag id");
    static_assert(hsm::weights_tag.is_pointer(), "weights_tag must always be pointer-mode");

    static_assert(hsm::compiled_options_tag.id() == hsm::compiled_options, "compiled_options_tag id");
    static_assert(hsm::compiled_options_tag.is_pointer(), "compiled_options_tag must always be pointer-mode");

    SUCCEED();
}

TEST_F(HsmFormatLayoutCompatibilityTest, device_tags_cannot_collide_with_core_tags) {
    // A device tag built from local id 0 must never land in the Core-owned range, no matter how small the
    // local id is - this is the whole point of SectionTag::make_device_tag() vs. picking a raw absolute id by hand.
    static_assert(hsm::SectionTag::make_device_tag(0, true).device_local_id() == 0,
                  "make_device_tag()/device_local_id() must round-trip the local id");
    static_assert(hsm::SectionTag::make_device_tag(0, true).id() == hsm::core_tag_id_range_end,
                  "make_device_tag(0, ...) must land exactly at the range boundary");
    static_assert(hsm::SectionTag::make_device_tag(0, true).id() >= hsm::core_tag_id_range_end,
                  "device tag ids must never fall below core_tag_id_range_end");
    static_assert(hsm::model_id < hsm::core_tag_id_range_end, "model_id must stay below core_tag_id_range_end");
    static_assert(hsm::model < hsm::core_tag_id_range_end, "model must stay below core_tag_id_range_end");

    SUCCEED();
}

TEST_F(HsmFormatLayoutCompatibilityTest, core_tag_values_are_pinned) {
    static_assert(static_cast<uint32_t>(hsm::Tags::invalid) == 0, "Tags::invalid value changed");
    static_assert(static_cast<uint32_t>(hsm::Tags::model_id) == 1, "Tags::model_id value changed");
    static_assert(static_cast<uint32_t>(hsm::Tags::model) == 2, "Tags::model value changed");
    static_assert(static_cast<uint32_t>(hsm::Tags::runtime_requirements) == 3,
                  "Tags::runtime_requirements value changed");
    static_assert(static_cast<uint32_t>(hsm::Tags::model_struct) == 4, "Tags::model_struct value changed");
    static_assert(static_cast<uint32_t>(hsm::Tags::weights) == 5, "Tags::weights value changed");
    static_assert(static_cast<uint32_t>(hsm::Tags::compiled_options) == 6, "Tags::compiled_options value changed");

    SUCCEED();
}

TEST_F(HsmFormatLayoutCompatibilityTest, core_tag_ids_cannot_be_shared) {
    static_assert(all_unique(k_core_tag_ids), "two Core tags resolve to the same wire id");
    static_assert(all_in_range(k_core_tag_ids, 1u, hsm::core_tag_id_range_end),
                  "a Core tag id reuses Tags::invalid or escapes the Core-owned range");
    // +1 for Tags::invalid, which is not a real tag and so is absent from k_core_tag_ids.
    static_assert(k_core_tag_ids.size() + 1 == static_cast<size_t>(hsm::Tags::sentinel_count),
                  "a Core tag was added to Tags but not listed in k_core_tag_ids");

    SUCCEED();
}

TEST_F(HsmFormatLayoutCompatibilityTest, device_values_are_pinned_and_cannot_be_shared) {
    static_assert(static_cast<hsm::DeviceId>(hsm::Devices::any) == 0, "Devices::any value changed");
    static_assert(static_cast<hsm::DeviceId>(hsm::Devices::cpu) == 1, "Devices::cpu value changed");
    static_assert(static_cast<hsm::DeviceId>(hsm::Devices::gpu) == 2, "Devices::gpu value changed");
    static_assert(static_cast<hsm::DeviceId>(hsm::Devices::npu) == 3, "Devices::npu value changed");

    static_assert(all_unique(k_device_ids), "two devices resolve to the same wire id");
    static_assert(all_in_range(k_device_ids, hsm::any_device_id, hsm::sample_device_id_range_start),
                  "a real device id escapes into the sample/test bucket");
    static_assert(k_device_ids.size() == static_cast<size_t>(hsm::Devices::sentinel_count),
                  "a device was added to Devices but not listed in k_device_ids");

    SUCCEED();
}
// --- HSM wire-format layout/version compatibility end ----------------------------------------------------

TEST(HsmHeaderTest, check_single_blob_header) {
    const auto blob = make_sample_single_blob_container();
    ASSERT_GE(blob.size(), sizeof(hsm::Header));

    const auto header = hsm::Header::view(blob.data());
    EXPECT_EQ(header.magic, hsm::BlobMagic::single);
    EXPECT_EQ(header.version_major, hsm::FormatVersion::major);
    EXPECT_EQ(header.version_minor, hsm::FormatVersion::minor);
    EXPECT_EQ(header.container_size, blob.size());
    EXPECT_EQ(header.manifest_offset, sizeof(hsm::Header) + 2u);  // 2 bytes of section payload
    EXPECT_EQ(header.manifest_size, 2 * k_manifest_entry_size);   // 2 entries in the manifest
};

TEST(HsmHeaderTest, check_multi_blob_single_blob) {
    const auto blob = make_multi_blob_container(1);
    ASSERT_GE(blob.size(), sizeof(hsm::Header));

    const auto header = hsm::Header::view(blob.data());
    EXPECT_EQ(header.magic, hsm::BlobMagic::multi);
    EXPECT_EQ(header.version_major, hsm::FormatVersion::major);
    EXPECT_EQ(header.version_minor, hsm::FormatVersion::minor);
    EXPECT_EQ(header.container_size, blob.size());
    EXPECT_EQ(header.manifest_offset, sizeof(hsm::Header));
    EXPECT_EQ(header.manifest_size, 0u);  // no entries in the manifest
};

TEST(HsmHeaderTest, check_multi_blob_two_blobs) {
    const auto blob = make_multi_blob_container(2);
    ASSERT_GE(blob.size(), 2 * sizeof(hsm::Header));

    const auto header1 = hsm::Header::view(blob.data());
    EXPECT_EQ(header1.magic, hsm::BlobMagic::multi);
    EXPECT_EQ(header1.version_major, hsm::FormatVersion::major);
    EXPECT_EQ(header1.version_minor, hsm::FormatVersion::minor);
    EXPECT_EQ(header1.container_size, sizeof(hsm::Header));
    EXPECT_EQ(header1.manifest_offset, sizeof(hsm::Header));
    EXPECT_EQ(header1.manifest_size, 0u);  // no entries in the manifest

    const auto header2 = hsm::Header::view(blob.data() + header1.container_size);
    EXPECT_EQ(header2.magic, hsm::BlobMagic::multi);
    EXPECT_EQ(header2.version_major, hsm::FormatVersion::major);
    EXPECT_EQ(header2.version_minor, hsm::FormatVersion::minor);
    EXPECT_EQ(header2.container_size, sizeof(hsm::Header));
    EXPECT_EQ(header2.manifest_offset, sizeof(hsm::Header));
    EXPECT_EQ(header2.manifest_size, 0u);  // no entries in the manifest
}

TEST(HsmContainerViewTest, reads_header) {
    const auto blob = make_sample_container_with_entries();
    const hsm::ContainerView view(blob.data(), blob.size());
    EXPECT_EQ(view.size(), blob.size());
    EXPECT_EQ(view.header().magic, hsm::BlobMagic::single);
}

TEST(HsmContainerViewTest, reads_manifest_entries) {
    const auto blob = make_sample_container_with_entries();
    const hsm::ContainerView view(blob.data(), blob.size());

    ASSERT_EQ(view.manifest_count(), 2u);
    const auto* manifest = &view.manifest();

    const auto& id_entry = manifest[0];
    EXPECT_EQ(id_entry.tag.id(), hsm::model_id);
    EXPECT_TRUE(id_entry.tag.is_inline());
    EXPECT_EQ(id_entry.inline_bytes[0], 0xAA);

    const auto& model_entry = manifest[1];
    EXPECT_EQ(model_entry.tag.id(), hsm::model);
    EXPECT_TRUE(model_entry.tag.is_pointer());
    EXPECT_EQ(model_entry.size, 2u);
}

TEST(HsmContainerViewTest, reads_pointer_section) {
    const auto blob = make_sample_container_with_entries();
    const hsm::ContainerView view(blob.data(), blob.size());
    const auto& model_entry = (&view.manifest())[1];
    const auto section = view.section(model_entry);
    ASSERT_EQ(section.size(), 2u);
    EXPECT_EQ(std::string(reinterpret_cast<const char*>(section.data()), section.size()), "OV");
}

TEST(HsmContainerViewTest, section_rejects_inline_mode_entry) {
    const auto blob = make_sample_container_with_entries();
    const hsm::ContainerView view(blob.data(), blob.size());

    hsm::ManifestEntry entry{};
    entry.tag = hsm::model_id_tag;  // inline-mode: offset/size below don't refer to a real section
    entry.offset = sizeof(hsm::Header);
    entry.size = 2;

    EXPECT_EQ(view.section(entry).size(), 0u);
}

TEST(HsmContainerViewTest, section_rejects_out_of_bounds_offset) {
    const auto blob = make_sample_container_with_entries();
    const hsm::ContainerView view(blob.data(), blob.size());

    hsm::ManifestEntry entry{};
    entry.tag = hsm::model_tag;
    entry.offset = view.size() + 1;
    entry.size = 1;

    EXPECT_EQ(view.section(entry).size(), 0u);
}

TEST(HsmContainerViewTest, section_rejects_out_of_bounds_size) {
    const auto blob = make_sample_container_with_entries();
    const hsm::ContainerView view(blob.data(), blob.size());

    hsm::ManifestEntry entry{};
    entry.tag = hsm::model_tag;
    entry.offset = 0;
    entry.size = view.size() + 1;  // fits at offset 0 alone, but overruns the buffer

    EXPECT_EQ(view.section(entry).size(), 0u);
}

TEST(HsmContainerViewValidateTest, accepts_well_formed_container) {
    const auto blob = make_sample_container_with_entries();
    const hsm::ContainerView view(blob.data(), blob.size());
    EXPECT_TRUE(view.validate());
}

TEST(HsmContainerViewValidateTest, accepts_well_formed_multi_container) {
    const auto blob = make_multi_container({}, {});
    const hsm::ContainerView view(blob.data(), blob.size());
    EXPECT_TRUE(view.validate());
}

TEST(HsmContainerViewValidateTest, rejects_bad_magic) {
    auto blob = make_sample_container_with_entries();
    blob[0] = 'X';  // corrupt Header::magic
    const hsm::ContainerView view(blob.data(), blob.size());
    EXPECT_FALSE(view.validate());
}

TEST(HsmContainerViewValidateTest, rejects_mismatched_major_version) {
    auto blob = make_sample_container_with_entries();
    auto header = hsm::Header::view(blob.data());
    header.version_major = hsm::FormatVersion::major + 1;
    std::memcpy(blob.data(), &header, sizeof(header));

    const hsm::ContainerView view(blob.data(), blob.size());
    EXPECT_FALSE(view.validate());
}

TEST(HsmContainerViewValidateTest, rejects_buffer_smaller_than_total_size) {
    const auto blob = make_sample_container_with_entries();
    // View sees fewer bytes than Header::container_size claims.
    const hsm::ContainerView view(blob.data(), blob.size() - 1);
    EXPECT_FALSE(view.validate());
}

TEST(HsmContainerViewValidateTest, rejects_out_of_bounds_manifest_offset) {
    auto blob = make_sample_container_with_entries();
    auto header = hsm::Header::view(blob.data());
    header.manifest_offset = std::numeric_limits<hsm::OffsetType>::max();
    std::memcpy(blob.data(), &header, sizeof(header));

    const hsm::ContainerView view(blob.data(), blob.size());
    EXPECT_FALSE(view.validate());
}

TEST(HsmContainerViewValidateTest, rejects_manifest_offset_inside_header) {
    auto blob = make_sample_container_with_entries();
    auto header = hsm::Header::view(blob.data());
    header.manifest_offset = sizeof(hsm::Header) - 1;  // would start reading manifest inside the header
    std::memcpy(blob.data(), &header, sizeof(header));

    const hsm::ContainerView view(blob.data(), blob.size());
    EXPECT_FALSE(view.validate());
}

TEST(HsmContainerViewValidateTest, rejects_manifest_size_not_multiple_of_entry_size) {
    auto blob = make_sample_container_with_entries();
    auto header = hsm::Header::view(blob.data());
    header.manifest_size -= 1;  // no longer a multiple of sizeof(ManifestEntry)
    std::memcpy(blob.data(), &header, sizeof(header));

    const hsm::ContainerView view(blob.data(), blob.size());
    EXPECT_FALSE(view.validate());
}

TEST(HsmContainerViewValidateTest, accepts_container_with_empty_manifest) {
    const auto blob = make_single_blob_container({}, {});  // no section payload, no manifest entries
    const hsm::ContainerView view(blob.data(), blob.size());
    ASSERT_EQ(view.manifest_count(), 0u);
    EXPECT_TRUE(view.validate());
}

TEST(HsmContainerViewValidateTest, rejects_out_of_bounds_section_offset) {
    auto blob = make_sample_container_with_entries();
    const auto header = hsm::Header::view(blob.data());

    // Second manifest entry is the pointer-mode "model" tag.
    const auto entry_offset = header.manifest_offset + sizeof(hsm::ManifestEntry);
    hsm::ManifestEntry entry{};
    std::memcpy(&entry, blob.data() + entry_offset, sizeof(entry));
    entry.offset = std::numeric_limits<hsm::OffsetType>::max();
    std::memcpy(blob.data() + entry_offset, &entry, sizeof(entry));

    const hsm::ContainerView view(blob.data(), blob.size());
    EXPECT_FALSE(view.validate());
}

TEST(HsmContainerViewValidateTest, rejects_section_overlapping_header) {
    auto blob = make_sample_container_with_entries();
    const auto header = hsm::Header::view(blob.data());

    // Second manifest entry is the pointer-mode "model" tag; point it at the header itself.
    const auto entry_offset = header.manifest_offset + sizeof(hsm::ManifestEntry);
    hsm::ManifestEntry entry{};
    std::memcpy(&entry, blob.data() + entry_offset, sizeof(entry));
    entry.offset = 0;
    entry.size = 4;
    std::memcpy(blob.data() + entry_offset, &entry, sizeof(entry));

    const hsm::ContainerView view(blob.data(), blob.size());
    EXPECT_FALSE(view.validate());
}

TEST(HsmMultiBlobViewTest, empty_buffer_has_no_blobs) {
    const hsm::MultiBlobView view(static_cast<const uint8_t*>(nullptr), 0);
    EXPECT_EQ(view.blob_count(), 0u);
}

TEST(HsmMultiBlobViewTest, reads_single_blob) {
    const auto blob = make_multi_blob_file(1);
    const hsm::MultiBlobView view(blob.data(), blob.size());

    ASSERT_EQ(view.blob_count(), 1u);
    EXPECT_EQ(view.blob_at(0).header().magic, hsm::BlobMagic::single);
}

TEST(HsmMultiBlobViewTest, reads_multiple_blobs) {
    const auto blob = make_multi_blob_file(3);
    const hsm::MultiBlobView view(blob.data(), blob.size());

    ASSERT_EQ(view.blob_count(), 3u);
    for (size_t i = 0; i < view.blob_count(); ++i) {
        EXPECT_EQ(view.blob_at(i).header().magic, hsm::BlobMagic::single);
    }
}

TEST(HsmMultiBlobViewTest, skips_optional_shared_context_between_blobs) {
    auto buffer = make_multi_blob_file(1);                    // mandatory shared context + 1 blob
    const auto extra_context = make_multi_container({}, {});  // optional shared-context update
    buffer.insert(buffer.end(), extra_context.begin(), extra_context.end());
    const auto blob1 = make_single_blob_container({}, {});
    buffer.insert(buffer.end(), blob1.begin(), blob1.end());

    const hsm::MultiBlobView view(buffer.data(), buffer.size());
    ASSERT_EQ(view.blob_count(), 2u);  // the extra shared-context container doesn't count as a blob
    EXPECT_EQ(view.blob_at(0).header().magic, hsm::BlobMagic::single);
    EXPECT_TRUE(view.blob_at(0).validate());
    EXPECT_EQ(view.blob_at(1).header().magic, hsm::BlobMagic::single);
}

TEST(HsmMultiBlobViewTest, stops_on_oversized_total_size) {
    auto blob = make_multi_blob_file(1);
    auto header = hsm::Header::view(blob.data());  // corrupt the mandatory shared context's header
    header.container_size = std::numeric_limits<hsm::SizeType>::max();
    std::memcpy(blob.data(), &header, sizeof(header));

    const hsm::MultiBlobView view(blob.data(), blob.size());
    EXPECT_EQ(view.blob_count(), 0u);       // can't even get past the corrupt first container
    EXPECT_EQ(view.blob_at(0).size(), 0u);  // out-of-range -> empty view, not a crash
}

TEST(HsmMultiBlobViewTest, stops_on_invalid_magic) {
    auto blob = make_multi_blob_file(1);
    const auto second_container_offset = sizeof(hsm::Header);
    blob[second_container_offset] = 'X';  // corrupt the blob's magic to neither single nor multi

    const hsm::MultiBlobView view(blob.data(), blob.size());
    EXPECT_EQ(view.blob_count(), 0u);  // shared context is skipped fine, but the blob itself is unreadable
}

TEST(HsmMultiBlobViewTest, blob_view_excludes_following_containers) {
    auto buffer = make_multi_container({}, {});  // mandatory shared context
    const auto blob0 = make_single_blob_container({}, {});
    const auto blob1 = make_single_blob_container({'O', 'V'}, {});  // different size than blob0
    buffer.insert(buffer.end(), blob0.begin(), blob0.end());
    buffer.insert(buffer.end(), blob1.begin(), blob1.end());

    const hsm::MultiBlobView view(buffer.data(), buffer.size());
    ASSERT_EQ(view.blob_count(), 2u);

    const auto view0 = view.blob_at(0);
    EXPECT_EQ(view0.size(), blob0.size());  // must not leak into blob1's bytes
    EXPECT_TRUE(view0.validate());
}

TEST(HsmMultiBlobViewTest, stops_on_mismatched_major_version) {
    auto buffer = make_multi_container({}, {});  // mandatory shared context
    auto header = hsm::Header::view(buffer.data());
    header.version_major = hsm::FormatVersion::major + 1;
    std::memcpy(buffer.data(), &header, sizeof(header));
    const auto blob0 = make_single_blob_container({}, {});
    buffer.insert(buffer.end(), blob0.begin(), blob0.end());

    const hsm::MultiBlobView view(buffer.data(), buffer.size());
    EXPECT_EQ(view.blob_count(), 0u);  // can't trust framing past an unsupported major version
}

TEST(IHsmSectionExtensionTest, recognizes_own_device_and_tag) {
    hsm::ManifestEntry entry{};
    entry.device = RecordingExtension::owned_device;
    entry.tag = hsm::SectionTag::make(RecordingExtension::owned_tag_id, /*is_inline=*/false);

    const std::byte payload[4] = {};
    RecordingExtension extension;
    EXPECT_TRUE(extension.read_section(entry, ov::util::MemoryView{payload, 4}));
    EXPECT_EQ(extension.last_section_size, 4u);
}

TEST(IHsmSectionExtensionTest, skips_entry_it_does_not_own) {
    hsm::ManifestEntry entry{};
    entry.device = RecordingExtension::owned_device + 1;  // different device
    entry.tag = hsm::SectionTag::make(RecordingExtension::owned_tag_id, /*is_inline=*/false);

    RecordingExtension extension;
    EXPECT_FALSE(extension.read_section(entry, ov::util::MemoryView{}));
    EXPECT_EQ(extension.last_section_size, 0u);  // never called
}

// OPENVINO_DEBUG_ASSERT compiles out entirely under NDEBUG (Release builds), so this only runs in debug builds.
#ifndef NDEBUG
TEST(MakeDeviceTagTest, debug_asserts_on_id_overflow) {
    const auto out_of_range_id = hsm::max_tag_id - hsm::core_tag_id_range_end + 1;
    EXPECT_DEATH(hsm::SectionTag::make_device_tag(out_of_range_id, false), "");
}
#endif

}  // namespace ov::test
