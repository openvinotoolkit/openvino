// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_deferred_writer.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <utility>
#include <variant>

#include "openvino/core/except.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/variant_visitor.hpp"

namespace ov::runtime::hsm {
inline namespace v1 {

namespace {

constexpr size_t k_inline_capacity = sizeof(ManifestEntry{}.inline_bytes);

size_t normalize_alignment(size_t alignment) {
    const size_t a = alignment == 0 ? 1 : alignment;
    const auto is_power_of_two = (a & (a - 1)) == 0;
    OPENVINO_ASSERT(is_power_of_two, "HSM section alignment must be a power of two");
    return a;
}

SectionAlignment resolve_alignment(SectionAlignment align) {
    const auto off_align = normalize_alignment(align.offset_align);
    const auto slot_align = align.size_align == 0 ? off_align : normalize_alignment(align.size_align);
    return {off_align, slot_align};
}

// Reserved byte range for one section - computed once so the size dry run and the actual write pass
// in DeferredWriter::finalize() can never disagree about where anything ends up.
struct SectionSlot {
    size_t start;
    size_t end;
};

constexpr SectionSlot reserve_slot(size_t cursor, size_t offset_align, size_t size_align, size_t size) {
    const auto start = ov::util::align_size_up(cursor, offset_align);
    return {start, start + ov::util::align_size_up(size, size_align)};
}

}  // namespace

std::optional<size_t> DeferredWriter::reserve(BufferDestination& destination, size_t extra) {
    if (!destination.good || destination.size + extra > destination.capacity) {
        destination.good = false;
        return std::nullopt;
    } else {
        const auto at = destination.size;
        destination.size += extra;
        return at;
    }
}

void DeferredWriter::write_into(StreamDestination& destination, ov::util::MemoryView data) {
    destination.stream->write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    destination.size += data.size();
}

void DeferredWriter::write_into(BufferDestination& destination, ov::util::MemoryView data) {
    if (const auto at = reserve(destination, data.size())) {
        std::memcpy(destination.dst + *at, data.data(), data.size());
    }
}

void DeferredWriter::write(ov::util::MemoryView data) {
    std::visit(
        [&](auto& destination) {
            write_into(destination, data);
        },
        m_destination);
}

void DeferredWriter::write_zeros(size_t count) {
    static constexpr std::array<std::byte, 256> zeros{};
    size_t remaining = count;
    while (remaining > 0) {
        const size_t chunk = std::min(remaining, zeros.size());
        write({zeros.data(), chunk});
        remaining -= chunk;
    }
}

void DeferredWriter::patch_into(StreamDestination& destination, size_t offset, ov::util::MemoryView data) {
    const auto resume = destination.stream->tellp();
    destination.stream->seekp(static_cast<std::streamoff>(offset));
    destination.stream->write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    destination.stream->seekp(resume);
}

void DeferredWriter::patch_into(BufferDestination& destination, size_t offset, ov::util::MemoryView data) {
    OPENVINO_ASSERT(destination.good && offset + data.size() <= destination.size, "HSM writer: patch out of bounds");
    std::memcpy(destination.dst + offset, data.data(), data.size());
}

void DeferredWriter::patch(size_t offset, ov::util::MemoryView data) {
    std::visit(
        [&](auto& destination) {
            patch_into(destination, offset, data);
        },
        m_destination);
}

size_t DeferredWriter::written_size() const {
    return std::visit(
        [](const auto& d) {
            return d.size;
        },
        m_destination);
}

bool DeferredWriter::is_good(const StreamDestination& destination) {
    return static_cast<bool>(*destination.stream);
}

bool DeferredWriter::is_good(const BufferDestination& destination) {
    return destination.good;
}

bool DeferredWriter::destination_good() const {
    return std::visit(
        [](const auto& destination) {
            return is_good(destination);
        },
        m_destination);
}

// Writes one pointer- or inline-mode section from an already-available view, producing its manifest entry.
ManifestEntry DeferredWriter::write_view_section(DeviceId device,
                                                 SectionTag tag,
                                                 size_t offset_align,
                                                 size_t size_align,
                                                 ov::util::MemoryView payload) {
    ManifestEntry entry{};
    entry.device = device;
    entry.tag = tag;
    if (tag.is_inline()) {
        std::memcpy(entry.inline_bytes.data(), payload.data(), payload.size());
        return entry;
    }
    const auto slot = reserve_slot(written_size(), offset_align, size_align, payload.size());
    write_zeros(slot.start - written_size());  // pad to the aligned start
    if (destination_good()) {
        write(payload);
    }
    if (destination_good()) {
        write_zeros(slot.end - written_size());  // pad the slot
    }
    entry.offset = slot.start;
    entry.size = payload.size();
    return entry;
}

// Same as above, but the payload is generated on demand by @p encode - always pointer-mode.
ManifestEntry DeferredWriter::write_encoded_section(DeviceId device,
                                                    SectionTag tag,
                                                    size_t offset_align,
                                                    size_t size_align,
                                                    size_t size,
                                                    const SectionEncoder& encode) {
    ManifestEntry entry{};
    entry.device = device;
    entry.tag = tag;
    const auto slot = reserve_slot(written_size(), offset_align, size_align, size);
    write_zeros(slot.start - written_size());
    if (destination_good()) {
        size_t remaining = size;
        const SectionSink sink = [this, &remaining](ov::util::MemoryView data) {
            OPENVINO_ASSERT(data.size() <= remaining, "HSM SectionEncoder wrote past its declared size");
            remaining -= data.size();
            write(data);
        };
        encode(sink);
        // A destination that ran out of room mid-encode already failed for an unrelated reason - don't
        // also flag that as a broken encoder.
        OPENVINO_ASSERT(!destination_good() || remaining == 0,
                        "HSM SectionEncoder must append exactly its declared size");
    }
    if (destination_good()) {
        write_zeros(slot.end - written_size());
    }
    entry.offset = slot.start;
    entry.size = size;
    return entry;
}

// Same as write_encoded_section(), but @p encode's output length isn't known until it returns - only the
// aligned start can be computed up front; the slot's end (and the manifest entry's size) are measured
// from how far the destination actually grew.
ManifestEntry DeferredWriter::write_open_encoded_section(DeviceId device,
                                                         SectionTag tag,
                                                         size_t offset_align,
                                                         size_t size_align,
                                                         const SectionEncoder& encode) {
    ManifestEntry entry{};
    entry.device = device;
    entry.tag = tag;
    const size_t start = ov::util::align_size_up(written_size(), offset_align);
    write_zeros(start - written_size());
    if (destination_good()) {
        const SectionSink sink = [this](ov::util::MemoryView data) {
            write(data);
        };
        encode(sink);
    }
    const size_t real_size = destination_good() ? written_size() - start : 0;
    if (destination_good()) {
        write_zeros(ov::util::align_size_up(real_size, size_align) - real_size);
    }
    entry.offset = start;
    entry.size = real_size;
    return entry;
}

DeferredWriter::DeferredWriter(std::ostream& stream) noexcept
    : m_destination(StreamDestination{&stream}),
      m_sections(),
      m_finalized(false),
      m_has_unsized_section(false) {}

DeferredWriter::DeferredWriter(std::byte* dst, size_t capacity) noexcept
    : m_destination(BufferDestination{dst, capacity}),
      m_sections(),
      m_finalized(false),
      m_has_unsized_section(false) {}

std::optional<DeferredWriter> DeferredWriter::open(std::ostream& stream) {
    DeferredWriter writer{stream};
    return writer.destination_good() ? std::optional<DeferredWriter>{std::move(writer)} : std::nullopt;
}

std::optional<DeferredWriter> DeferredWriter::open(std::byte* dst, size_t size) {
    if (dst == nullptr || size < sizeof(Header)) {
        return std::nullopt;
    }
    DeferredWriter writer{dst, size};
    return writer.destination_good() ? std::optional<DeferredWriter>{std::move(writer)} : std::nullopt;
}

bool DeferredWriter::add_section(DeviceId device,
                                 SectionTag tag,
                                 ov::util::MemoryView payload,
                                 SectionAlignment align) {
    const auto resolved = resolve_alignment(align);
    OPENVINO_ASSERT(!tag.is_inline() || payload.size() <= k_inline_capacity,
                    "HSM inline section payload exceeds the manifest entry's inline capacity");
    m_sections.push_back({resolved, payload, device, tag});
    return true;
}

bool DeferredWriter::add_section(DeviceId device,
                                 SectionTag tag,
                                 size_t size,
                                 SectionEncoder encode,
                                 SectionAlignment align) {
    OPENVINO_ASSERT(!tag.is_inline(), "HSM fill-in-place sections must be pointer-mode");
    const auto resolved = resolve_alignment(align);
    m_sections.push_back({resolved, PendingEncode{size, std::move(encode)}, device, tag});
    return true;
}

bool DeferredWriter::add_section(DeviceId device, SectionTag tag, SectionEncoder encode, SectionAlignment align) {
    OPENVINO_ASSERT(!tag.is_inline(), "HSM fill-in-place sections must be pointer-mode");
    const auto resolved = resolve_alignment(align);
    m_sections.push_back({resolved, std::move(encode), device, tag});
    m_has_unsized_section = true;
    return true;
}

std::error_code DeferredWriter::finalize() {
    if (!m_finalized) {
        m_finalized = true;
        const auto section_count = m_sections.size();
        size_t manifest_offset = 0;
        if (m_has_unsized_section) {
            // At least one section's size is only known after encoding it, so the header can't be
            // computed up front - write a placeholder and patch it once every real size is known.
            write_zeros(sizeof(Header));
        } else {
            // Every section's size is already known (nothing is written until finalize()), so the header
            // can be computed up front, in a pure pass with no I/O, and written first - no seek-back needed.
            const auto payload_size = [](const PendingSection& section) -> size_t {
                return std::visit(ov::util::VariantVisitor{
                                      [](const ov::util::MemoryView& payload) {
                                          return payload.size();
                                      },
                                      [](const PendingEncode& payload) {
                                          return payload.size;
                                      },
                                      [](const auto&) -> size_t {
                                          OPENVINO_THROW("unreachable: m_has_unsized_section guards this path");
                                      },
                                  },
                                  section.payload);
            };

            auto body_size = sizeof(Header);
            for (const auto& section : m_sections) {
                if (!section.tag.is_inline()) {
                    body_size = reserve_slot(body_size,
                                             section.align.offset_align,
                                             section.align.size_align,
                                             payload_size(section))
                                    .end;
                }
            }

            Header header{};
            header.magic = BlobMagic::single;
            header.version_major = FormatVersion::major;
            header.version_minor = FormatVersion::minor;
            header.manifest_offset = body_size;
            header.manifest_size = section_count * sizeof(ManifestEntry);
            header.container_size = body_size + header.manifest_size;
            write({reinterpret_cast<const std::byte*>(&header), sizeof(header)});
            manifest_offset = body_size;
        }

        std::vector<ManifestEntry> entries(section_count);
        for (size_t i = 0; i < section_count && destination_good(); ++i) {
            const auto& section = m_sections[i];
            entries[i] = std::visit(ov::util::VariantVisitor{
                                        [&](const ov::util::MemoryView& payload) -> ManifestEntry {
                                            return write_view_section(section.device,
                                                                      section.tag,
                                                                      section.align.offset_align,
                                                                      section.align.size_align,
                                                                      payload);
                                        },
                                        [&](const PendingEncode& payload) -> ManifestEntry {
                                            return write_encoded_section(section.device,
                                                                         section.tag,
                                                                         section.align.offset_align,
                                                                         section.align.size_align,
                                                                         payload.size,
                                                                         payload.encode);
                                        },
                                        [&](const SectionEncoder& encode) -> ManifestEntry {
                                            return write_open_encoded_section(section.device,
                                                                              section.tag,
                                                                              section.align.offset_align,
                                                                              section.align.size_align,
                                                                              encode);
                                        },
                                    },
                                    section.payload);
        }

        if (m_has_unsized_section) {
            manifest_offset = written_size();  // now known: right before the manifest is written
        }
        if (destination_good()) {
            for (const auto& entry : entries) {
                write({reinterpret_cast<const std::byte*>(&entry), sizeof(entry)});
            }
        }
        if (m_has_unsized_section && destination_good()) {
            Header header{};
            header.magic = BlobMagic::single;
            header.version_major = FormatVersion::major;
            header.version_minor = FormatVersion::minor;
            header.manifest_offset = manifest_offset;
            header.manifest_size = entries.size() * sizeof(ManifestEntry);
            header.container_size = written_size();
            patch(0, {reinterpret_cast<const std::byte*>(&header), sizeof(header)});
        }
    }
    return destination_good() ? std::error_code{} : make_error_code(WriteErrc::write_failed);
}

}  // namespace v1
}  // namespace ov::runtime::hsm
