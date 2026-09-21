// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <istream>
#include <memory>
#include <optional>
#include <vector>

#include "openvino/runtime/hsm_format.hpp"
#include "openvino/runtime/shared_buffer.hpp"

namespace ov::runtime {
inline namespace v1 {

/**
 * @brief Common decoder contract: given a section's raw bytes, returns the decoded `T`, or `std::nullopt`
 * if the bytes don't parse.
 * @note Convention, not enforced: report malformed section bytes via `std::nullopt`, never throw.
 */
template <typename T>
using SectionDecoder = std::function<std::optional<T>(ov::util::MemoryView)>;

/**
 * @brief Common (device-agnostic) reader for a single HSM container, over any `std::istream&`.
 *
 * Validates the container and drives extension dispatch so plugins can consume - or override - any
 * section, without this reader ever needing to interpret proprietary content itself.
 *
 * A plain buffer or an mmap'd file is adapted to a stream via #ov::SharedStreamBuffer (see the
 * `open(data, size)` overloads), so there's one validation/reading path regardless of the source. Every
 * section accessor has a zero-copy `_view()` sibling - see #section_view() for when it applies.
 *
 * To turn a section's raw bytes into a plugin- or Core-owned type, pair any `section*()` result with
 * a #SectionDecoder - deliberately not a method here, so this reader stays uninvolved in what that type is.
 */
class HsmReader {
public:
    /**
     * @brief Opens an HSM container from a given input stream.
     *
     * @param stream The input stream containing the HSM container data. The stream must remain valid
     *               for the lifetime of the returned HsmReader.
     * @return std::optional<HsmReader> Returns a valid HsmReader if the container is successfully opened
     *                                  and validated; otherwise, returns std::nullopt.
     */
    static std::optional<HsmReader> open(std::istream& stream);

    /**
     * @brief Opens an HSM container from a given in-memory buffer.
     *
     * @param data Pointer to the buffer containing the HSM container data.
     * @param size Size of the buffer in bytes.
     * @return std::optional<HsmReader> Returns a valid HsmReader if the container is successfully opened
     *                                  and validated; otherwise, returns std::nullopt.
     */
    static std::optional<HsmReader> open(const std::byte* data, size_t size);

    /**
     * @brief Opens an HSM container from a given in-memory buffer (uint8_t variant).
     *
     * @param data Pointer to the buffer containing the HSM container data.
     * @param size Size of the buffer in bytes.
     * @return std::optional<HsmReader> Returns a valid HsmReader if the container is successfully opened
     *                                  and validated; otherwise, returns std::nullopt.
     */
    static std::optional<HsmReader> open(const uint8_t* data, size_t size);

    HsmReader(const HsmReader&) = delete;
    HsmReader& operator=(const HsmReader&) = delete;
    HsmReader(HsmReader&&) = default;
    HsmReader& operator=(HsmReader&&) = default;

    /**
     * @brief Returns the validated HSM header read at open() time.
     */
    const HSMHeader& header() const noexcept {
        return m_header;
    }

    /**
     * @brief Checks if the HSM container uses a shared context (multi-blob) layout.
     * @return true if the container is shared context (#BlobMagic::multi), false otherwise (#BlobMagic::single).
     */
    bool is_shared_context() const noexcept {
        return m_header.magic == BlobMagic::multi;
    }

    /**
     * @brief Finds the first manifest entry matching (@p device, @p tag) and returns its payload, regardless
     * of whether the entry is pointer- or inline-mode. @p tag may be a raw `uint32_t` id or any enum-typed
     * tag (#HSMTags, or a plugin's own device-specific tag enum) - no manual `static_cast` needed.
     * @note Nothing in the format guarantees a (device, tag) pair appears at most once - use
     * #sections() if more than one entry may legitimately share the same (device, tag).
     */
    template <typename Tag>
    std::optional<std::vector<std::byte>> section(DeviceId device, Tag tag) const {
        return section_by_id(device, static_cast<uint32_t>(tag));
    }

    /// @overload Uses #any_device_id - shorthand for Core sections; no new accessor needed as tags grow.
    template <typename Tag>
    std::optional<std::vector<std::byte>> section(Tag tag) const {
        return section_by_id(any_device_id, static_cast<uint32_t>(tag));
    }

    /**
     * @brief Finds all manifest entries matching (@p device, @p tag) and returns their payloads, in manifest order.
     * @note Use this when multiple entries may legitimately share the same (device, tag).
     */
    template <typename Tag>
    std::vector<std::vector<std::byte>> sections(DeviceId device, Tag tag) const {
        return sections_by_id(device, static_cast<uint32_t>(tag));
    }

    /**
     * @brief Zero-copy view of the first manifest entry matching (@p device, @p tag), if available.
     * @note Always available for inline-mode entries (payload lives in the manifest entry itself); for pointer-mode,
     * only when opened over an addressable buffer (`open(data, size)`) - `std::nullopt` otherwise.
     */
    template <typename Tag>
    std::optional<ov::util::MemoryView> section_view(DeviceId device, Tag tag) const noexcept {
        return section_view_by_id(device, static_cast<uint32_t>(tag));
    }

    /// @overload Uses #any_device_id - shorthand for a Core-owned section.
    template <typename Tag>
    std::optional<ov::util::MemoryView> section_view(Tag tag) const noexcept {
        return section_view_by_id(any_device_id, static_cast<uint32_t>(tag));
    }

    /**
     * @brief Zero-copy sibling of #sections() - entries that can't be viewed without a copy are omitted.
     */
    template <typename Tag>
    std::vector<ov::util::MemoryView> sections_view(DeviceId device, Tag tag) const {
        return sections_view_by_id(device, static_cast<uint32_t>(tag));
    }

    /**
     * @brief Dispatches every manifest entry - Core-owned sections included - to @p extensions, in order,
     * stopping at the first one that recognizes it. An entry no extension recognizes is silently skipped.
     * @return Number of entries actually handled by some extension.
     */
    size_t read_sections(const std::vector<IHsmSectionExtension*>& extensions) const;

private:
    /**
     * @brief Owns the SharedStreamBuffer+istream pair adapting a raw buffer, for the open(data,size) overloads.
     */
    struct OwnedStream {
        ov::SharedStreamBuffer buffer;
        std::istream stream;
        OwnedStream(const std::byte* data, size_t size) : buffer(data, size), stream(&buffer) {}
    };

    /**
     * @brief Set only for the open(data,size) overloads: owns the adapter stream and views the same bytes for
     * zero-copy `_view()` accessors - always set/unset together, so one member covers both.
     */
    struct BufferSource {
        std::unique_ptr<OwnedStream> owned;
        ov::util::MemoryView buffer;
    };

    HsmReader(std::istream& stream,
              std::optional<BufferSource> buffer_source,
              std::streampos start,
              HSMHeader header,
              std::vector<ManifestEntry> manifest) noexcept
        : m_buffer_source(std::move(buffer_source)),
          m_stream(stream),
          m_start(start),
          m_header(header),
          m_manifest(std::move(manifest)) {}

    static std::optional<HsmReader> open_stream(std::istream& stream, std::optional<BufferSource> buffer_source);
    std::optional<std::vector<std::byte>> section_payload(const ManifestEntry& entry) const;
    std::optional<ov::util::MemoryView> section_view(const ManifestEntry& entry) const noexcept;

    std::optional<std::vector<std::byte>> section_by_id(DeviceId device, uint32_t tag_id) const;
    std::vector<std::vector<std::byte>> sections_by_id(DeviceId device, uint32_t tag_id) const;
    std::optional<ov::util::MemoryView> section_view_by_id(DeviceId device, uint32_t tag_id) const noexcept;
    std::vector<ov::util::MemoryView> sections_view_by_id(DeviceId device, uint32_t tag_id) const;

    std::optional<BufferSource> m_buffer_source;    //!< set only by open(data,size) - enables zero-copy views
    std::reference_wrapper<std::istream> m_stream;  //!< reseatable (unlike a bare reference) - move-assignable
    std::streampos m_start;                         //!< Start position of the container within the stream.
    HSMHeader m_header{};                           //!< Parsed HSM header.
    std::vector<ManifestEntry> m_manifest;          //!< Parsed manifest entries.
};

}  // namespace v1
}  // namespace ov::runtime
