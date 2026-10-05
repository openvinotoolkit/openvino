// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_deferred_writer.hpp"

#include <algorithm>
#include <array>
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

constexpr SectionSlot reserve_slot(size_t cursor, size_t size, SectionAlignment align) {
    const auto start = ov::util::align_size_up(cursor, align.offset_align);
    return {start, start + ov::util::align_size_up(size, align.size_align)};
}

}  // namespace

std::optional<size_t> DeferredWriter::reserve(BufferDestination& destination, size_t extra) {
    OPENVINO_ASSERT(destination.size <= destination.capacity, "HSM writer: buffer size exceeds its capacity");
    if (!destination.good || extra > destination.capacity - destination.size) {
        destination.good = false;
        return std::nullopt;
    } else {
        return std::exchange(destination.size, destination.size + extra);
    }
}

DeferredWriter::PendingSection::PendingSection(SectionAlignment align,
                                               ov::util::MemoryView payload,
                                               DeviceId device,
                                               SectionTag tag)
    : align{align},
      payload{payload},
      device{device},
      tag{tag} {}

DeferredWriter::PendingSection::PendingSection(SectionAlignment align,
                                               PendingEncode payload,
                                               DeviceId device,
                                               SectionTag tag)
    : align{align},
      payload{std::move(payload)},
      device{device},
      tag{tag} {}

DeferredWriter::PendingSection::PendingSection(SectionAlignment align,
                                               SectionEncoder payload,
                                               DeviceId device,
                                               SectionTag tag)
    : align{align},
      payload{std::move(payload)},
      device{device},
      tag{tag} {}

void DeferredWriter::write_into(StreamDestination& destination, ov::util::MemoryView data) {
    destination.stream->write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    destination.size += data.size();
}

void DeferredWriter::write_into(BufferDestination& destination, ov::util::MemoryView data) {
    if (const auto at = reserve(destination, data.size())) {
        std::copy_n(data.data(), data.size(), destination.dst + *at);
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
    while (remaining > 0 && destination_good()) {
        const size_t chunk = std::min(remaining, zeros.size());
        write({zeros.data(), chunk});
        remaining -= chunk;
    }
}

std::streampos DeferredWriter::container_start(const StreamDestination& destination) {
    return destination.stream->tellp() - static_cast<std::streamoff>(destination.size);
}

void DeferredWriter::patch_into(StreamDestination& destination, size_t offset, ov::util::MemoryView data) {
    const auto resume = destination.stream->tellp();
    destination.stream->seekp(container_start(destination) + static_cast<std::streamoff>(offset));
    destination.stream->write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    destination.stream->seekp(resume);
}

void DeferredWriter::patch_into(BufferDestination& destination, size_t offset, ov::util::MemoryView data) {
    OPENVINO_ASSERT(destination.good && offset + data.size() <= destination.size, "HSM writer: patch out of bounds");
    std::copy_n(data.data(), data.size(), destination.dst + offset);
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

void DeferredWriter::reset_destination() {
    std::visit(ov::util::VariantVisitor{
                   [](StreamDestination& destination) {
                       if (destination.size != 0) {
                           // Nothing written yet on a fresh writer - a seek would needlessly require a
                           // seekable stream even for a container made only of sized sections.
                           destination.stream->seekp(container_start(destination));
                           destination.size = 0;
                       }
                   },
                   [](BufferDestination& destination) {
                       destination.size = 0;
                       destination.good = true;
                   },
               },
               m_destination);
}

ManifestEntry DeferredWriter::write_section(DeviceId device,
                                            SectionTag tag,
                                            SectionAlignment align,
                                            ov::util::MemoryView payload) {
    ManifestEntry entry{};
    entry.device = device;
    entry.tag = tag;
    if (tag.is_inline()) {
        std::copy_n(reinterpret_cast<const uint8_t*>(payload.data()), payload.size(), entry.inline_bytes.data());
        return entry;
    }
    const auto slot = reserve_slot(written_size(), payload.size(), align);
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

ManifestEntry DeferredWriter::write_section(DeviceId device,
                                            SectionTag tag,
                                            SectionAlignment align,
                                            const PendingEncode& payload) {
    ManifestEntry entry{};
    entry.device = device;
    entry.tag = tag;
    const auto slot = reserve_slot(written_size(), payload.size, align);
    write_zeros(slot.start - written_size());
    if (destination_good()) {
        size_t remaining = payload.size;
        const SectionSink sink = [this, &remaining](ov::util::MemoryView data) {
            OPENVINO_ASSERT(data.size() <= remaining, "HSM SectionEncoder wrote past its declared size");
            remaining -= data.size();
            write(data);
        };
        payload.encode(sink);
        // A destination that ran out of room mid-encode already failed for an unrelated reason - don't
        // also flag that as a broken encoder.
        OPENVINO_ASSERT(!destination_good() || remaining == 0,
                        "HSM SectionEncoder must append exactly its declared size");
    }
    if (destination_good()) {
        write_zeros(slot.end - written_size());
    }
    entry.offset = slot.start;
    entry.size = payload.size;
    return entry;
}

ManifestEntry DeferredWriter::write_section(DeviceId device,
                                            SectionTag tag,
                                            SectionAlignment align,
                                            const SectionEncoder& encode) {
    ManifestEntry entry{};
    entry.device = device;
    entry.tag = tag;
    const size_t start = ov::util::align_size_up(written_size(), align.offset_align);
    write_zeros(start - written_size());
    if (destination_good()) {
        const SectionSink sink = [this](ov::util::MemoryView data) {
            write(data);
        };
        encode(sink);
    }
    const size_t real_size = destination_good() ? written_size() - start : 0;
    if (destination_good()) {
        write_zeros(ov::util::align_size_up(real_size, align.size_align) - real_size);
    }
    entry.offset = start;
    entry.size = real_size;
    return entry;
}

DeferredWriter::DeferredWriter(std::ostream& stream) noexcept
    : m_destination{StreamDestination{&stream}},
      m_sections{},
      m_result{},
      m_has_unsized_section{false} {}

DeferredWriter::DeferredWriter(std::byte* dst, size_t capacity) noexcept
    : m_destination{BufferDestination{dst, capacity}},
      m_sections{},
      m_result{},
      m_has_unsized_section{false} {}

std::optional<DeferredWriter> DeferredWriter::open(std::ostream& stream) {
    DeferredWriter writer{stream};
    return writer.destination_good() ? std::optional<DeferredWriter>{std::move(writer)} : std::nullopt;
}

std::optional<DeferredWriter> DeferredWriter::open(std::byte* dst, size_t size) {
    if (dst == nullptr || size < sizeof(Header)) {
        return std::nullopt;
    } else {
        DeferredWriter writer{dst, size};
        return writer.destination_good() ? std::optional<DeferredWriter>{std::move(writer)} : std::nullopt;
    }
}

bool DeferredWriter::add_section(DeviceId device,
                                 SectionTag tag,
                                 ov::util::MemoryView payload,
                                 SectionAlignment align) {
    const auto resolved = resolve_alignment(align);
    OPENVINO_ASSERT(!tag.is_inline() || payload.size() <= k_inline_capacity,
                    "HSM inline section payload exceeds the manifest entry's inline capacity");
    m_sections.emplace_back(resolved, payload, device, tag);
    return true;
}

bool DeferredWriter::add_section(DeviceId device,
                                 SectionTag tag,
                                 size_t size,
                                 SectionEncoder encode,
                                 SectionAlignment align) {
    OPENVINO_ASSERT(!tag.is_inline(), "HSM fill-in-place sections must be pointer-mode");
    const auto resolved = resolve_alignment(align);
    m_sections.emplace_back(resolved, PendingEncode{size, std::move(encode)}, device, tag);
    return true;
}

bool DeferredWriter::add_section(DeviceId device, SectionTag tag, SectionEncoder encode, SectionAlignment align) {
    OPENVINO_ASSERT(!tag.is_inline(), "HSM fill-in-place sections must be pointer-mode");
    const auto resolved = resolve_alignment(align);
    m_sections.emplace_back(resolved, std::move(encode), device, tag);
    m_has_unsized_section = true;
    return true;
}

std::error_code DeferredWriter::finalize() {
    if (!m_result.has_value()) {
        reset_destination();
        const auto section_count = m_sections.size();
        size_t manifest_offset = 0;
        if (m_has_unsized_section) {
            // At least one section's size is only known after encoding it, so the header can't be
            // computed up front - write a placeholder and patch it once every real size is known.
            write_zeros(sizeof(Header));
        } else {
            // std::get is safe here - m_has_unsized_section already rules out a SectionEncoder payload.
            const auto payload_size = [](const PendingSection& section) -> size_t {
                if (const auto view = std::get_if<ov::util::MemoryView>(&section.payload)) {
                    return view->size();
                } else {
                    return std::get<PendingEncode>(section.payload).size;
                }
            };

            auto body_size = sizeof(Header);
            for (const auto& section : m_sections) {
                if (!section.tag.is_inline()) {
                    body_size = reserve_slot(body_size, payload_size(section), section.align).end;
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
            entries[i] = std::visit(
                [&](const auto& payload) -> ManifestEntry {
                    return write_section(section.device, section.tag, section.align, payload);
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
        m_result = destination_good() ? std::error_code{} : make_error_code(WriteErrc::write_failed);
    }
    return *m_result;
}

}  // namespace v1
}  // namespace ov::runtime::hsm
