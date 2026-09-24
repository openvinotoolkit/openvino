// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/hsm_reader.hpp"

#include <algorithm>
#include <cstring>

namespace ov::runtime {
inline namespace v1 {

std::optional<HsmReader> HsmReader::open(std::istream& stream) {
    return open_stream(stream, std::nullopt);
}

std::optional<HsmReader> HsmReader::open(const std::byte* data, size_t size) {
    auto owned = std::make_unique<OwnedStream>(data, size);
    auto& stream = owned->stream;
    return open_stream(stream, BufferSource{std::move(owned), ov::util::MemoryView{data, size}});
}

std::optional<HsmReader> HsmReader::open(const uint8_t* data, size_t size) {
    return open(reinterpret_cast<const std::byte*>(data), size);
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
    }
    stream.seekg(start + static_cast<std::streamoff>(header.manifest_offset));
    return static_cast<bool>(
        stream.read(reinterpret_cast<char*>(manifest.data()), static_cast<std::streamsize>(header.manifest_size)));
}
}  // namespace

// Shared by every open() overload; `buffer_source`, when set, owns the SharedStreamBuffer+istream pair and
// views the same bytes `stream` reads from - enables the zero-copy `_view()` accessors.
std::optional<HsmReader> HsmReader::open_stream(std::istream& stream, std::optional<BufferSource> buffer_source) {
    const auto start = stream.tellg();
    HSMHeader header{};
    std::vector<ManifestEntry> manifest;

    if (start == std::streampos(-1) || !stream.read(reinterpret_cast<char*>(&header), sizeof(header)) ||
        !is_valid_header_fields(header) || !read_manifest(stream, start, header, manifest)) {
        return std::nullopt;
    }

    const bool valid = buffer_source
                           ? HSMContainerView{buffer_source->buffer.data(), buffer_source->buffer.size()}.validate()
                           : std::all_of(manifest.begin(), manifest.end(), [&header](const ManifestEntry& entry) {
                                 return is_valid_section_bounds(entry, header);
                             });

    return valid ? std::optional<HsmReader>(
                       HsmReader{stream, std::move(buffer_source), start, header, std::move(manifest)})
                 : std::nullopt;
}

std::optional<std::vector<std::byte>> HsmReader::section_payload(const ManifestEntry& entry) const {
    if (entry.tag.is_inline()) {
        std::vector<std::byte> bytes(entry.inline_bytes.size());
        std::memcpy(bytes.data(), entry.inline_bytes.data(), entry.inline_bytes.size());
        return bytes;
    } else {
        std::vector<std::byte> bytes(static_cast<size_t>(entry.size));
        m_stream.get().seekg(m_start + static_cast<std::streamoff>(entry.offset));
        if (m_stream.get().read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(entry.size))) {
            return bytes;
        } else {
            return std::nullopt;
        }
    }
}

std::optional<ov::util::MemoryView> HsmReader::section_view(const ManifestEntry& entry) const noexcept {
    if (entry.tag.is_inline()) {
        // Lives inside the manifest entry itself (owned by this HsmReader) - always viewable, no copy.
        return ov::util::MemoryView{reinterpret_cast<const std::byte*>(entry.inline_bytes.data()),
                                    entry.inline_bytes.size()};
    } else if (m_buffer_source) {
        // Bounds already validated at open() time (see open_stream()).
        return ov::util::MemoryView{m_buffer_source->buffer.data() + static_cast<size_t>(entry.offset),
                                    static_cast<size_t>(entry.size)};
    } else {
        return std::nullopt;
    }
}

std::optional<std::vector<std::byte>> HsmReader::section_by_id(DeviceId device, uint32_t tag_id) const {
    const auto found = std::find_if(m_manifest.begin(), m_manifest.end(), [device, tag_id](const ManifestEntry& entry) {
        return entry_matches(entry, device, tag_id);
    });
    return found != m_manifest.end() ? section_payload(*found) : std::nullopt;
}

std::vector<std::vector<std::byte>> HsmReader::sections_by_id(DeviceId device, uint32_t tag_id) const {
    std::vector<std::vector<std::byte>> result;
    for (const auto& entry : m_manifest) {
        if (entry_matches(entry, device, tag_id)) {
            if (auto payload = section_payload(entry)) {
                result.push_back(std::move(*payload));
            }
        }
    }
    return result;
}

std::optional<ov::util::MemoryView> HsmReader::section_view_by_id(DeviceId device, uint32_t tag_id) const noexcept {
    const auto found = std::find_if(m_manifest.begin(), m_manifest.end(), [device, tag_id](const ManifestEntry& entry) {
        return entry_matches(entry, device, tag_id);
    });
    return found != m_manifest.end() ? section_view(*found) : std::nullopt;
}

std::vector<ov::util::MemoryView> HsmReader::sections_view_by_id(DeviceId device, uint32_t tag_id) const {
    std::vector<ov::util::MemoryView> result;
    for (const auto& entry : m_manifest) {
        if (entry_matches(entry, device, tag_id)) {
            if (auto view = section_view(entry)) {
                result.push_back(*view);
            }
        }
    }
    return result;
}

size_t HsmReader::read_sections(const std::vector<IHsmSectionExtension*>& extensions) const {
    size_t handled = 0;
    for (const auto& entry : m_manifest) {
        std::optional<std::vector<std::byte>> owned;  // only materialized when a zero-copy view isn't possible
        ov::util::MemoryView view;
        if (const auto direct = section_view(entry)) {
            view = *direct;
        } else {
            owned = section_payload(entry);
            if (!owned) {
                continue;
            }
            view = {owned->data(), owned->size()};
        }
        for (auto* extension : extensions) {
            if (extension != nullptr && extension->read_section(entry, view)) {
                ++handled;
                break;
            }
        }
    }
    return handled;
}

}  // namespace v1
}  // namespace ov::runtime
