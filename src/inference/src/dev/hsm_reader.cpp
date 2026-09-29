// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_reader.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

#include "openvino/util/variant_visitor.hpp"

namespace ov::runtime {
inline namespace v1 {

std::optional<ov::util::MemoryView> HsmSection::view() const noexcept {
    if (m_entry.tag.is_inline()) {
        return ov::util::MemoryView{reinterpret_cast<const std::byte*>(m_entry.inline_bytes.data()),
                                    m_entry.inline_bytes.size()};
    } else {
        return std::visit(
            ov::util::VariantVisitor{[](const auto&) -> std::optional<ov::util::MemoryView> {
                                         return std::nullopt;
                                     },
                                     [](const ov::util::MemoryView& payload) -> std::optional<ov::util::MemoryView> {
                                         return payload;
                                     }},
            m_source);
    }
}

bool HsmSection::read(size_t offset, size_t length, void* destination) const {
    if (const auto total = size(); destination == nullptr || offset > total || length > total - offset) {
        return false;
    }

    if (m_entry.tag.is_inline()) {
        std::memcpy(destination, m_entry.inline_bytes.data() + offset, length);
        return true;
    } else {
        return std::visit(ov::util::VariantVisitor{
                              [](const auto&) {
                                  return false;
                              },
                              [&](const ov::util::MemoryView& payload) {
                                  std::memcpy(destination, payload.data() + offset, length);
                                  return true;
                              },
                              [&](const StreamSource& stream_source) {
                                  auto& [stream, payload_start] = stream_source;
                                  stream->seekg(payload_start + static_cast<std::streamoff>(offset));
                                  return static_cast<bool>(stream->read(reinterpret_cast<char*>(destination),
                                                                        static_cast<std::streamsize>(length)));
                              },
                          },
                          m_source);
    }
}

namespace {
bool entry_matches(const ManifestEntry& entry, DeviceId device, uint32_t tag_id) noexcept {
    return entry.device == device && entry.tag.id() == tag_id;
}

// Resizes `manifest` per header.manifest_size and reads it in; a no-op read for an empty manifest.
bool read_manifest(std::istream& stream,
                   std::streampos start,
                   const HSMHeader& header,
                   std::vector<ManifestEntry>& manifest) {
    manifest.resize(static_cast<size_t>(header.manifest_size / sizeof(ManifestEntry)));
    if (manifest.empty()) {
        return true;
    } else {
        stream.seekg(start + static_cast<std::streamoff>(header.manifest_offset));
        return static_cast<bool>(
            stream.read(reinterpret_cast<char*>(manifest.data()), static_cast<std::streamsize>(header.manifest_size)));
    }
}

bool has_container_size_bytes(std::istream& stream, std::streampos start, const HSMHeader& header) {
    stream.seekg(start);
    stream.ignore(static_cast<std::streamsize>(header.container_size));
    return stream.good();
}

// 32-bit size_t can't represent every HSMSizeType value - reject oversized sections before they're constructed.
constexpr bool section_size_fits_size_t(const ManifestEntry& entry) noexcept {
    if constexpr (sizeof(size_t) < sizeof(HSMSizeType)) {
        return !entry.tag.is_pointer() || entry.size <= static_cast<HSMSizeType>(std::numeric_limits<size_t>::max());
    } else {
        return true;
    }
}
}  // namespace

std::optional<HsmReader> HsmReader::open(std::istream& stream) {
    const auto start = stream.tellg();
    HSMHeader header{};
    std::vector<ManifestEntry> manifest;

    if (start == std::streampos(-1) || !stream.read(reinterpret_cast<char*>(&header), sizeof(header)) ||
        !is_valid_header_fields(header) || !read_manifest(stream, start, header, manifest) ||
        !has_container_size_bytes(stream, start, header) ||
        !std::all_of(manifest.begin(), manifest.end(), [&header](const ManifestEntry& entry) {
            return is_valid_section_bounds(entry, header) && section_size_fits_size_t(entry);
        })) {
        return std::nullopt;
    } else {
        return HsmReader{std::nullopt, &stream, start, std::move(manifest)};
    }
}

std::optional<HsmReader> HsmReader::open(const std::byte* data, size_t size) {
    const HSMContainerView view{data, size};
    if (!view.validate()) {
        return std::nullopt;
    }
    std::vector<ManifestEntry> manifest(view.manifest_count());
    if (!manifest.empty()) {
        std::memcpy(manifest.data(), &view.manifest(), manifest.size() * sizeof(ManifestEntry));
    }
    return HsmReader{ov::util::MemoryView{data, size}, nullptr, std::streampos(0), std::move(manifest)};
}

std::optional<HsmReader> HsmReader::open(const uint8_t* data, size_t size) {
    return open(reinterpret_cast<const std::byte*>(data), size);
}

HsmSection HsmReader::make_section(const ManifestEntry& entry) const {
    if (entry.tag.is_inline()) {
        return HsmSection{entry};
    } else {
        return m_buffer ? HsmSection{entry,
                                     ov::util::MemoryView{m_buffer->data() + static_cast<size_t>(entry.offset),
                                                          static_cast<size_t>(entry.size)}}
                        : HsmSection{entry, *m_stream, m_start + static_cast<std::streamoff>(entry.offset)};
    }
}

std::optional<HsmSection> HsmReader::section_by_id(DeviceId device, uint32_t tag_id) const {
    const auto found = std::find_if(m_manifest.begin(), m_manifest.end(), [device, tag_id](const auto& entry) {
        return entry_matches(entry, device, tag_id);
    });
    return found != m_manifest.end() ? std::optional<HsmSection>(make_section(*found)) : std::nullopt;
}

std::vector<HsmSection> HsmReader::sections_by_id(DeviceId device, uint32_t tag_id) const {
    std::vector<HsmSection> result;
    for (const auto& entry : m_manifest) {
        if (entry_matches(entry, device, tag_id)) {
            result.push_back(make_section(entry));
        }
    }
    return result;
}

size_t HsmReader::count_by_id(DeviceId device, uint32_t tag_id) const noexcept {
    return static_cast<size_t>(std::count_if(m_manifest.begin(), m_manifest.end(), [device, tag_id](const auto& entry) {
        return entry_matches(entry, device, tag_id);
    }));
}

size_t HsmReader::read_sections(const std::vector<IHsmSectionHandler*>& handlers) const {
    size_t handled = 0;
    for (const auto& entry : m_manifest) {
        const auto section = make_section(entry);
        std::optional<std::vector<std::byte>> owned;  // only materialized when a zero-copy view isn't possible
        ov::util::MemoryView view;
        if (const auto direct = section.view()) {
            view = *direct;
        } else {
            owned = section.to_bytes();
            if (!owned) {
                continue;
            }
            view = {owned->data(), owned->size()};
        }
        for (const auto& handler : handlers) {
            if (handler != nullptr && handler->handle_section(entry, view)) {
                ++handled;
                break;
            }
        }
    }
    return handled;
}

}  // namespace v1
}  // namespace ov::runtime
