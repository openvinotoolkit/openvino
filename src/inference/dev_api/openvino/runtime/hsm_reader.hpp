// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <istream>
#include <optional>
#include <variant>
#include <vector>

#include "openvino/runtime/hsm_format.hpp"
#include "openvino/runtime/common.hpp"

namespace ov::runtime {
inline namespace v1 {

/**
 * @brief Copyable handle to one manifest entry's payload - the single way to read a #HsmSection, whether the
 * owning #HsmReader was opened over a stream or a buffer. Never owns or copies memory itself; #read() is
 * the only copy point.
 * @note Outlives the #HsmReader that produced it, as long as the stream/buffer passed to `open()` is still
 * alive.
 */
class OPENVINO_RUNTIME_API HsmSection {
public:
    /**
     * @brief Inline-mode section - payload lives in @p entry itself, no external source needed.
     */
    explicit constexpr HsmSection(const ManifestEntry& entry) noexcept : m_entry(entry) {}

    /**
     * @brief Pointer-mode section backed by an addressable buffer; @p payload is its own bytes only.
     */
    constexpr HsmSection(const ManifestEntry& entry, ov::util::MemoryView payload) noexcept
        : m_entry(entry),
          m_source(payload) {}

    /**
     * @brief Pointer-mode section backed by a stream; @p payload_start is its first byte's stream position.
     */
    HsmSection(const ManifestEntry& entry, std::istream& stream, std::streampos payload_start) noexcept
        : m_entry(entry),
          m_source(StreamSource{&stream, payload_start}) {}

    constexpr const ManifestEntry& entry() const noexcept {
        return m_entry;
    }
    constexpr DeviceId device() const noexcept {
        return m_entry.device;
    }
    constexpr SectionTag tag() const noexcept {
        return m_entry.tag;
    }
    constexpr size_t size() const noexcept {
        return m_entry.tag.is_inline() ? m_entry.inline_bytes.size() : static_cast<size_t>(m_entry.size);
    }

    /**
     * @brief Zero-copy view of the whole section.
     * @return `std::nullopt` only for a pointer-mode section read from a stream-only #HsmReader - there's no
     * addressable memory to view; use #read() instead. Slice the returned view yourself for a sub-range.
     */
    std::optional<ov::util::MemoryView> view() const noexcept;

    /**
     * @brief Reads `[offset, offset + length)` into caller-owned @p destination - the only copy this type
     * ever makes. Always works, regardless of source; prefer #view() first to avoid the copy when possible.
     * @param destination Must point to at least @p length writable bytes, kept alive by the caller - `void*`
     * so any buffer type (`ov::Tensor::data()`, `vector<uint8_t>::data()`, ...) can be passed without a cast.
     * @return false if `[offset, offset + length)` exceeds this section's size, or the underlying read failed.
     */
    bool read(size_t offset, size_t length, void* destination) const;

    /**
     * @overload Reads the whole section; @p destination must point to at least #size() writable bytes.
     */
    bool read(void* destination) const {
        return read(0, size(), destination);
    }

    /**
     * @brief Owned-copy convenience - explicit opt-in, since #HsmSection itself never copies on its own.
     * @return `std::nullopt` if the underlying #read() failed.
     */
    std::optional<std::vector<std::byte>> to_bytes() const {
        std::vector<std::byte> bytes(size());
        return read(bytes.data()) ? std::optional(std::move(bytes)) : std::nullopt;
    }

private:
    using StreamSource = std::pair<std::istream*, std::streampos>;

    ManifestEntry m_entry;
    std::variant<std::monostate, ov::util::MemoryView, StreamSource> m_source;
};

/**
 * @brief Common decoder contract: given a section, returns the decoded `T`, or `std::nullopt` if it doesn't
 * parse. Written once, works unchanged whether #HsmSection was opened over a stream or a buffer, and whether
 * the decoder reads all of it (#HsmSection::read()/#HsmSection::view()) or only part, decided from what it already
 * read - the reader itself never interprets content, so nothing here needs to change per source or per shape
 * of read.
 * @note Convention, not enforced: report malformed content via `std::nullopt`, never throw.
 * @note Distinct from #IHsmSectionHandler: this is a stateless function for one already-known `(device, tag)`
 * pair, invoked via #HsmReader::decode()/#HsmReader::decode_all(). #IHsmSectionHandler is for a set of
 * stateful handlers dispatched across an entire manifest in one pass, without enumerating tags up front.
 */
template <typename T>
using HsmSectionDecoder = std::function<std::optional<T>(const HsmSection&)>;

/**
 * @brief Adapts @p parse - the actual bytes-to-`T` logic, written once against a #ov::util::MemoryView -
 * into a #HsmSectionDecoder<T>. Runs @p parse directly over #HsmSection::view() when one is available (no extra
 * copy beyond what @p parse itself does), otherwise falls back to #HsmSection::read() into an owned buffer and
 * runs @p parse over that instead (exactly one copy) - so @p parse never needs to know or care which one
 * happened.
 */
template <typename T>
HsmSectionDecoder<T> make_hsm_section_decoder(std::function<std::optional<T>(ov::util::MemoryView)> parse) {
    return [parse = std::move(parse)](const HsmSection& section) -> std::optional<T> {
        if (const auto view = section.view()) {
            return parse(*view);
        } else {
            const auto bytes = section.to_bytes();
            return bytes ? parse(ov::util::MemoryView(bytes->data(), bytes->size())) : std::nullopt;
        }
    };
}

/**
 * @brief Common (device-agnostic) reader for a single HSM container, over any `std::istream&` or addressable
 * buffer. Validates the container and drives extension dispatch so plugins can consume - or override - any
 * section, without ever needing to interpret proprietary content itself.
 * @note Deliberately stateless beyond the parsed manifest - every accessor resolves fresh by `(device, tag)`,
 * so an #HsmReader is cheap to move. #entries() gives a no-payload-read overview; #section()/#sections() hand
 * back #HsmSection handles; #decode()/#decode_all() run a #HsmSectionDecoder over them directly.
 */
class OPENVINO_RUNTIME_API HsmReader {
public:
    /**
     * @brief Opens an HSM container from a given input stream.
     * @param stream The input stream containing the HSM container data. Must remain valid for the lifetime
     * of the returned HsmReader, and of any #HsmSection obtained from it.
     * @return A valid HsmReader if the container is successfully opened and validated; `std::nullopt` otherwise.
     */
    static std::optional<HsmReader> open(std::istream& stream);

    /**
     * @brief Opens an HSM container from a given in-memory buffer - enables zero-copy #HsmSection::view().
     * @param data Pointer to the buffer containing the HSM container data. Must remain valid for the lifetime
     * of the returned HsmReader, and of any #HsmSection obtained from it.
     * @param size Size of the buffer in bytes.
     * @return A valid HsmReader if the container is successfully opened and validated; `std::nullopt` otherwise.
     */
    static std::optional<HsmReader> open(const std::byte* data, size_t size);

    /**
     * @overload uint8_t variant of open(const std::byte*, size_t).
     */
    static std::optional<HsmReader> open(const uint8_t* data, size_t size);

    HsmReader(const HsmReader&) = delete;
    HsmReader& operator=(const HsmReader&) = delete;
    HsmReader(HsmReader&&) = default;
    HsmReader& operator=(HsmReader&&) = default;

    /**
     * @brief Returns a reference to the vector of all manifest entries in the container.
     * @return A reference to the vector of all manifest entries in the container.
    */
    const std::vector<ManifestEntry>& entries() const noexcept {
        return m_manifest;
    }

    /**
     * @brief Number of manifest entries matching (@p device, @p tag) - no section payload is read.
     *
     * @param device The device ID to match.
     * @param tag The tag to match.
     * @return size_t The number of matching manifest entries.
     */
    template <typename Tag>
    size_t count(DeviceId device, Tag tag) const noexcept {
        return count_by_id(device, static_cast<uint32_t>(tag));
    }

    /**
     * @overload Uses #any_device_id - shorthand for a Core-owned section.
     */
    template <typename Tag>
    size_t count(Tag tag) const noexcept {
        return count_by_id(any_device_id, static_cast<uint32_t>(tag));
    }

    /**
     * @brief Finds the first manifest entry matching (@p device, @p tag) and returns a #HsmSection handle for it.
     *
     * @param device The device ID to match.
     * @param tag The tag to match.
     * @return std::optional<HsmSection> A handle to the matching section, or `std::nullopt` if not found.
     */
    template <typename Tag>
    std::optional<HsmSection> section(DeviceId device, Tag tag) const {
        return section_by_id(device, static_cast<uint32_t>(tag));
    }

    /**
     * @overload Uses #any_device_id - shorthand for Core sections; no new accessor needed as tags grow.
     */
    template <typename Tag>
    std::optional<HsmSection> section(Tag tag) const {
        return section_by_id(any_device_id, static_cast<uint32_t>(tag));
    }

    /**
     * @brief Finds all manifest entries matching (@p device, @p tag) and returns a #HsmSection handle for each, in manifest order.
     *
     * @tparam Tag The type of the tag, typically an enum or `uint32_t`.
     * @param device The device ID to match.
     * @param tag The tag to match.
     * @return std::vector<HsmSection> A vector of handles to the matching sections, in manifest order.
     */
    template <typename Tag>
    std::vector<HsmSection> sections(DeviceId device, Tag tag) const {
        return sections_by_id(device, static_cast<uint32_t>(tag));
    }

    /**
     * @overload Uses #any_device_id - shorthand for a Core-owned section.
     */
    template <typename Tag>
    std::vector<HsmSection> sections(Tag tag) const {
        return sections_by_id(any_device_id, static_cast<uint32_t>(tag));
    }

    /**
     * @brief Runs @p decoder over the first manifest entry matching (@p device, @p tag).
     * @note Returns std::nullopt if no matching entry is found, or if @p decoder itself returns std::nullopt.
     * @param device The device ID to match.
     * @param tag The tag to match.
     * @param decoder The decoder to run over the matching section.
     * @return A decoded value if the section is found and the decoder succeeds; otherwise, `std::nullopt`.
     */
    template <typename T, typename Tag>
    std::optional<T> decode(DeviceId device, Tag tag, const HsmSectionDecoder<T>& decoder) const {
        const auto found = section(device, tag);
        return found ? decoder(*found) : std::nullopt;
    }

    /**
     * @overload Uses #any_device_id - shorthand for a Core-owned section.
     */
    template <typename T, typename Tag>
    std::optional<T> decode(Tag tag, const HsmSectionDecoder<T>& decoder) const {
        return decode(any_device_id, tag, decoder);
    }

    /**
     * @brief Runs @p decoder over every manifest entry matching (@p device, @p tag) - entries @p decoder
     * itself rejects (returns std::nullopt for) are omitted, not treated as an error.
     * @note If no matching entries are found, an empty vector is returned.
     *
     * @param device The device ID to match.
     * @param tag The tag to match.
     * @param decoder The decoder to run over the matching sections.
     * @return A vector of decoded values for the matching sections that the decoder accepts.
     */
    template <typename T, typename Tag>
    std::vector<T> decode_all(DeviceId device, Tag tag, const HsmSectionDecoder<T>& decoder) const {
        std::vector<T> result;
        for (const auto& found : sections(device, tag)) {
            if (auto value = decoder(found)) {
                result.push_back(std::move(*value));
            }
        }
        return result;
    }

    /**
     * @overload Uses #any_device_id - shorthand for a Core-owned section.
     */
    template <typename T, typename Tag>
    std::vector<T> decode_all(Tag tag, const HsmSectionDecoder<T>& decoder) const {
        return decode_all(any_device_id, tag, decoder);
    }

    /**
     * @brief Dispatches every manifest entry - Core-owned sections included - to @p handlers, in order,
     * stopping at the first one that recognizes it. An entry no handler recognizes is silently skipped.
     *
     * @param handlers The list of handlers to dispatch the manifest entries to.
     * @return The number of entries actually handled by some handler.
     */
    size_t read_sections(const std::vector<IHsmSectionHandler*>& handlers) const;

private:
    HsmReader(std::optional<ov::util::MemoryView> buffer,
              std::istream* stream,
              std::streampos start,
              std::vector<ManifestEntry> manifest) noexcept
        : m_buffer(buffer),
          m_stream(stream),
          m_start(start),
          m_manifest(std::move(manifest)) {}

    HsmSection make_section(const ManifestEntry& entry) const;

    std::optional<HsmSection> section_by_id(DeviceId device, uint32_t tag_id) const;
    std::vector<HsmSection> sections_by_id(DeviceId device, uint32_t tag_id) const;
    size_t count_by_id(DeviceId device, uint32_t tag_id) const noexcept;

    std::optional<ov::util::MemoryView> m_buffer;  //!< set only by open(data,size)
    std::istream* m_stream;                        //!< non-owning; set only by open(stream)
    std::streampos m_start;                        //!< Start position of the container within #m_stream; else unused.
    std::vector<ManifestEntry> m_manifest;         //!< Parsed manifest entries.
};

}  // namespace v1
}  // namespace ov::runtime
