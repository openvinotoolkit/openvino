// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief HSM (Header-Sections-Manifest) blob container format contract.
 * @file openvino/runtime/hsm_format.hpp
 *
 * @verbatim
   +---------------------------+---------------------------+---------------------------+
   |         HSMHeader         |      Section payloads     |          Manifest         |
   |          32 bytes         |     variable size, 0+     | ManifestEntry[], 32B each |
   |          offset 0         |         offset 32         |  offset = manifest_offset |
   +---------------------------+---------------------------+---------------------------+
   @endverbatim
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "openvino/core/except.hpp"
#include "openvino/util/container_util.hpp"
#include "openvino/util/memory.hpp"

namespace ov::runtime {

using HSMMagicType = std::array<uint8_t, 5>;  //!< 5 raw ASCII bytes: container magic at offset 0.
using HSMSizeType = uint64_t;                 //!< Container/section size fields.
using HSMOffsetType = uint64_t;               //!< Byte offsets within the container.

/// Container magic (5 raw bytes).
struct BlobMagic {
    HSMMagicType value{};  //!< Raw bytes. Zero-initialized: an unset BlobMagic never matches a valid magic.

    /**
     * @brief Byte-wise compare; hand-written loop (not `value == other.value`) since
     * `std::array::operator==` isn't `constexpr` until C++20, and this must stay usable in a constant expression.
     */
    constexpr bool operator==(const BlobMagic& other) const noexcept {
        for (size_t i = 0; i < value.size(); ++i) {
            if (value[i] != other.value[i]) {
                return false;
            }
        }
        return true;
    }
    constexpr bool operator!=(const BlobMagic& other) const noexcept {
        return !(*this == other);
    }

    static const BlobMagic single;  //!< "OVBLS": container holds exactly one model blob.
    static const BlobMagic multi;   //!< "OVWSH": shared-context container - see #HSMMultiBlobView.
};

inline constexpr BlobMagic BlobMagic::single{ov::util::make_array<uint8_t>('O', 'V', 'B', 'L', 'S')};
inline constexpr BlobMagic BlobMagic::multi{ov::util::make_array<uint8_t>('O', 'V', 'W', 'S', 'H')};

/**
 * @brief The currently-active HSM v1.x format, as an `inline namespace`: `HSMHeader`, `HSMFormatVersion`,
 * `SectionTag`, `HSMTags`, `ManifestEntry` and friends all resolve here unqualified. A future major version
 * is a plain `namespace v2` alongside this one, explicit until promoted by moving `inline` here. Only
 * #BlobMagic and the basic type aliases above stay shared across every version.
 */
inline namespace v1 {

/**
 * @brief HSM format version written by this codebase.
 * @note Compatibility rule: a reader requires `header.version_major == major` exactly. #minor is additive-only
 * (new optional tags/fields never gate compatibility) - a reader must accept any #minor value once #major matches,
 * and must never fail solely on a #minor mismatch.
 */
struct HSMFormatVersion {
    static constexpr uint16_t major = 1;  //!< Major version written by this codebase.
    static constexpr uint8_t minor = 0;   //!< Minor version written by this codebase.
};

/**
 * @brief Fixed header at offset 0 of every HSM v1.x container. Compatibility lives in sections/manifest
 * entries, not in this header's layout.
 *
 * @verbatim
   +--------+------+-----------------+
   | Offset | Size | Field           |
   +--------+------+-----------------+
   | 0      | 5    | magic           |
   | 5      | 2    | version_major   |
   | 7      | 1    | version_minor   |
   | 8      | 8    | total_size      |
   | 16     | 8    | manifest_offset |
   | 24     | 8    | manifest_size   |
   +--------+------+-----------------+
   @endverbatim
 *
 * @note Native byte order (safe: all supported targets are little-endian); `#pragma pack(1)` keeps fields at
 * their documented offsets with no implicit padding.
 */
#pragma pack(push, 1)
struct HSMHeader {
    BlobMagic magic;         //!< Container magic - see #BlobMagic.
    uint16_t version_major;  //!< Format major version this container was written with - see #HSMFormatVersion.
    uint8_t version_minor;   //!< Format minor version this container was written with - see #HSMFormatVersion.

    HSMSizeType total_size;         //!< Whole container size, in bytes.
    HSMOffsetType manifest_offset;  //!< Byte offset of the first ManifestEntry.
    HSMSizeType manifest_size;      //!< Manifest size in bytes; entry count = manifest_size / sizeof(ManifestEntry).

    /**
     * @brief Non-owning view of the header at the start of an HSM container buffer - reinterprets the bytes
     * in place, no copy.
     * @warning No bounds checking: caller must ensure `data` points to at least `sizeof(HSMHeader)` readable bytes.
     */
    static const HSMHeader& view(const uint8_t* data) noexcept {
        return *reinterpret_cast<const HSMHeader*>(data);
    }
};
#pragma pack(pop)
static_assert(sizeof(HSMHeader) == 32,
              "HSMHeader layout changed - bump HSMFormatVersion::major/document the change before touching "
              "this struct, readers and writers (Story 1/2/3) must be updated together.");

/**
 * @brief Device/plugin that owns a ManifestEntry's section. Device catalog and collision rules follow later.
 * @note Paired with #SectionTag: a reader always compares `(device, tag)` together, never `tag` alone, so two
 * devices may reuse the same tag id for unrelated content.
 */
using DeviceId = uint8_t;

/**
 * @brief Semantic content of a ManifestEntry section (or inline payload); a raw 3-byte value split into a
 * 23-bit tag id (Core-owned range vs. device-specific range) plus a pointer/inline mode flag in bit 7 of
 * `value[0]` - see #is_inline().
 * @note Unknown-tag rule: a reader must skip any `(device, tag)` it doesn't recognize, never fail import.
 */
struct SectionTag {
    std::array<uint8_t, 3> value{};  //!< Raw bytes; bit 7 of `value[0]` is the pointer/inline flag.

    /**
     * @brief Packs a 23-bit tag id and the pointer/inline mode flag into a wire SectionTag.
     * @param id Tag id (Core-owned or device-specific range - see #SectionTag); only the low 23 bits are kept.
     * @param is_inline true for inline-mode, false for pointer-mode.
     */
    static constexpr SectionTag make(uint32_t id, bool is_inline) noexcept {
        return {{static_cast<uint8_t>(((id >> 16) & 0x7F) | (is_inline ? 0x80 : 0x00)),
                 static_cast<uint8_t>((id >> 8) & 0xFF),
                 static_cast<uint8_t>(id & 0xFF)}};
    }

    /// The 23-bit tag id (mode bit masked out).
    constexpr uint32_t id() const noexcept {
        return (static_cast<uint32_t>(value[0] & 0x7F) << 16) | (static_cast<uint32_t>(value[1]) << 8) |
               static_cast<uint32_t>(value[2]);
    }

    /// True if the entry carrying this tag stores its payload directly (bit 7 of `value[0]` set).
    constexpr bool is_inline() const noexcept {
        return (value[0] & 0x80) != 0;
    }

    /// True if the entry carrying this tag points at a section stored elsewhere.
    constexpr bool is_pointer() const noexcept {
        return !is_inline();
    }
};

/**
 * @brief Boundary in the 23-bit #SectionTag id space: Core ids are `< core_tag_id_range_end`;
 * #make_device_tag() only produces ids at/above it, so a device can never collide with a Core tag.
 */
inline constexpr uint32_t core_tag_id_range_end = 0x1000;

/// Highest id representable by the 23-bit #SectionTag id space.
inline constexpr uint32_t max_tag_id = 0x7FFFFF;

/**
 * @brief Core-owned tag ids. `invalid` (0) is reserved and never assigned to a real tag; mode is fixed per
 * tag (model id always inlines, model data never does) - use the `_tag()` factories below rather than
 * #SectionTag::make() so mode can't be set inconsistently.
 * @note Ids are auto-assigned via the enumerator order, so a new tag can never accidentally collide with
 * an existing one. `sentinel_count` is always one past the last real tag; the `static_assert` below guards
 * it against ever growing past #core_tag_id_range_end.
 */
enum class HSMTags : uint32_t {
    invalid = 0,           //!< Reserved: never a real tag id.
    model_id,              //!< See #model_id.
    model,                 //!< See #model.
    runtime_requirements,  //!< See #runtime_requirements_tag().
    // Add new Core tags above this line only - values are assigned automatically, in declaration order.
    sentinel_count,  // Not a real tag id - always exactly one past the last real entry above.
};
static_assert(static_cast<uint32_t>(HSMTags::sentinel_count) <= core_tag_id_range_end,
              "Too many Core tags defined for core_tag_id_range_end - widen the boundary (Story 4).");

inline constexpr uint32_t model_id = static_cast<uint32_t>(HSMTags::model_id);  //!< Model identifier (e.g. a hash).
inline constexpr uint32_t model = static_cast<uint32_t>(HSMTags::model);  //!< The serialized compiled model itself.
inline constexpr uint32_t runtime_requirements = static_cast<uint32_t>(HSMTags::runtime_requirements);

/// Wire tag for #model_id - always inline-mode.
constexpr SectionTag model_id_tag() noexcept {
    return SectionTag::make(model_id, /*is_inline=*/true);
}

/// Wire tag for #model - always pointer-mode.
constexpr SectionTag model_tag() noexcept {
    return SectionTag::make(model, /*is_inline=*/false);
}

/**
 * @brief Wire tag for #runtime_requirements - always pointer-mode. Payload is opaque to the common
 * reader/format: this contract only reserves the tag and its bounds (like any pointer-mode section) -
 * interpreting and enforcing the encoded requirements is entirely the emitting device/plugin's
 * responsibility, typically via #IHsmSectionExtension. No expression scheme is defined at this layer
 * (out of scope for Story 1; a richer format, if any, belongs to the tag registry - Story 5).
 */
constexpr SectionTag runtime_requirements_tag() noexcept {
    return SectionTag::make(runtime_requirements, /*is_inline=*/false);
}

/**
 * @brief Builds a device-specific wire #SectionTag from a device-local id; always lands at/above
 * #core_tag_id_range_end, so it can never collide with a Core tag.
 * @param local_id Device-local id, starting at 0; must be `<= max_tag_id - core_tag_id_range_end`
 * @param is_inline true for inline-mode, false for pointer-mode.
 */
constexpr SectionTag make_device_tag(uint32_t local_id, bool is_inline) noexcept {
    OPENVINO_DEBUG_ASSERT(local_id <= max_tag_id - core_tag_id_range_end);
    return SectionTag::make(core_tag_id_range_end + local_id, is_inline);
}

/**
 * @brief One fixed-size, 32-byte record of the manifest table (see #HSMHeader::manifest_offset).
 *
 * @verbatim
   +--------+------+-------------------------+
   | Offset | Size | Field                   |
   +--------+------+-------------------------+
   | 0      | 1    | device                  |
   | 1      | 3    | tag                     |
   | 4      | 4    | tag_reserved            |
   | 8      | 8    | offset                  |
   | 16     | 8    | size                    |
   | 24     | 8    | pointer_reserved        |
   +--------+------+-------------------------+
   Pointer-mode: offset/size/pointer_reserved are three named 8-byte fields (pointer_reserved must be 0).
   Inline-mode: the whole 24-byte region (offset 8-31) is reinterpreted as `inline_bytes` - up to 24 bytes
   embedded directly in the entry, no separate section payload. Which mode applies is read from
   #SectionTag::is_inline() - there is no separate mode byte.
   @endverbatim
 *
 * @note `#pragma pack(1)` is required: without it, the 3-byte `SectionTag` followed by the 8-byte-aligned
 * `offset` field would force padding, cascading misalignment through the rest of the struct.
 * @note A specific `(device, tag)` pair may redefine what its own `tag_reserved`/`pointer_reserved` bytes
 * mean; they're zero otherwise. This struct doesn't interpret content.
 */
#pragma pack(push, 1)
struct ManifestEntry {
    DeviceId device;                      //!< Owning device.
    SectionTag tag;                       //!< Section content;
    std::array<uint8_t, 4> tag_reserved;  //!< Zero, unless the specific (#device, #tag) pair redefines this.

    union {
        struct {
            HSMOffsetType offset;                     //!< Section payload offset (pointer-mode).
            HSMSizeType size;                         //!< Section payload size (pointer-mode).
            std::array<uint8_t, 8> pointer_reserved;  //!< Zero, unless (#device, #tag) redefines this.
        };
        std::array<uint8_t, 24> inline_bytes;  //!< Inline payload, up to 24 bytes (inline-mode).
    };
};
#pragma pack(pop)
static_assert(sizeof(ManifestEntry) == 32, "ManifestEntry layout changed.");

/**
 * @brief Reader-side plugin hook: interprets one manifest entry's section content. A concrete extension
 * self-dispatches by checking `(entry.device, entry.tag)` and returning whether it recognized it - per the
 * unknown-tag rule on #SectionTag, the reader must skip any entry no extension recognizes, never fail
 * import.
 * @note Forward-looking contract only (Story 1 scope) - not yet wired to a real reader (Story 2).
 */
class IHsmSectionExtension {
public:
    virtual ~IHsmSectionExtension() = default;

    /**
     * @brief Attempts to interpret @p entry's section content.
     * @param entry Manifest entry being considered - not necessarily one this extension owns.
     * @param section Bounds-checked view of the payload - #HSMContainerView::section() for a pointer-mode
     * entry, or `entry.inline_bytes` for an inline-mode one; never a raw, unchecked pointer.
     * @return true if `(entry.device, entry.tag)` was recognized and handled, false otherwise.
     */
    virtual bool read_section(const ManifestEntry& entry, ov::util::MemoryView section) = 0;
};

/**
 * @brief Read-only, zero-copy view of an entire in-memory HSM container: header, manifest and pointer-mode
 */
class HSMContainerView {
public:
    explicit constexpr HSMContainerView(const std::byte* data, size_t size) noexcept : m_view{data, size} {}
    explicit HSMContainerView(const uint8_t* data, size_t size) noexcept
        : HSMContainerView{reinterpret_cast<const std::byte*>(data), size} {}

    constexpr size_t size() const noexcept {
        return m_view.size();
    }

    /// Header at the start of the buffer.
    const HSMHeader& header() const noexcept {
        return HSMHeader::view(reinterpret_cast<const uint8_t*>(begin()));
    }

    /// First of `manifest_count()` entries at `header().manifest_offset`; use `&manifest()` for array access.
    const ManifestEntry& manifest() const noexcept {
        return *reinterpret_cast<const ManifestEntry*>(begin() + header().manifest_offset);
    }

    /// Number of entries at #manifest().
    size_t manifest_count() const noexcept {
        return static_cast<size_t>(header().manifest_size / sizeof(ManifestEntry));
    }

    /// Bounds-checked payload bytes of a pointer-mode manifest entry; empty view for an invalid or inline entry.
    constexpr ov::util::MemoryView section(const ManifestEntry& entry) const noexcept {
        if (!entry.tag.is_pointer() || entry.offset > size() || entry.size > size() - entry.offset) {
            return {};
        }
        return {begin() + static_cast<size_t>(entry.offset), static_cast<size_t>(entry.size)};
    }

    /**
     * @brief Basic structural integrity check: magic, and that the header/manifest/pointer-mode section
     * bounds all stay within #size() with no overflow. Doesn't interpret tag-specific (device, tag) content.
     */
    bool validate() const noexcept {
        if (static_cast<size_t>(end() - begin()) < sizeof(HSMHeader)) {
            return false;
        }
        const auto& hdr = header();
        if (hdr.magic != BlobMagic::single && hdr.magic != BlobMagic::multi) {
            return false;
        }
        if (hdr.version_major != HSMFormatVersion::major) {
            return false;
        }
        if (hdr.total_size < sizeof(HSMHeader) || hdr.total_size > size()) {
            return false;
        }
        if (hdr.manifest_size % sizeof(ManifestEntry) != 0) {
            return false;
        }
        if (hdr.manifest_offset < sizeof(HSMHeader) || hdr.manifest_offset > hdr.total_size ||
            hdr.total_size - hdr.manifest_offset < hdr.manifest_size) {
            return false;
        }
        if (hdr.manifest_size == 0) {
            return true;
        }
        const auto* entries = &manifest();
        for (size_t i = 0, count = manifest_count(); i < count; ++i) {
            const auto& entry = entries[i];
            if (entry.tag.is_pointer() && (entry.offset < sizeof(HSMHeader) || entry.offset > hdr.manifest_offset ||
                                           hdr.manifest_offset - entry.offset < entry.size)) {
                return false;
            }
        }
        return true;
    }

private:
    constexpr const std::byte* begin() const noexcept {
        return m_view.begin();
    }
    constexpr const std::byte* end() const noexcept {
        return m_view.end();
    }

    ov::util::MemoryView m_view;
};

/**
 * @brief View over a multi-blob HSM file: containers concatenated back-to-back, each self-describing via
 * its own `header().total_size`. Two kinds of container appear, disambiguated by magic: #BlobMagic::multi
 * ("shared context" - mandatory as the very first container, optional afterwards; carries data shared by
 * the blobs that follow it, until the next one) and #BlobMagic::single (an actual model blob). #blob_at()
 * only counts/returns the #BlobMagic::single containers - shared-context containers are skipped over.
 * Exact multi-blob framing is still evolving (Story 12/13).
 */
class HSMMultiBlobView {
public:
    explicit constexpr HSMMultiBlobView(const std::byte* data, size_t size) noexcept : m_view{data, size} {}
    explicit HSMMultiBlobView(const uint8_t* data, size_t size) noexcept
        : HSMMultiBlobView{reinterpret_cast<const std::byte*>(data), size} {}

    constexpr size_t size() const noexcept {
        return m_view.size();
    }

    /// Number of #BlobMagic::single containers; stops counting at the first invalid/out-of-bounds header.
    size_t blob_count() const noexcept {
        ov::util::MemoryView view = m_view;
        size_t count = 0;
        while (view.size() >= sizeof(HSMHeader)) {
            const auto next = advance_container(view);
            if (!next) {
                break;
            }
            count += next->is_blob ? 1 : 0;
            view = next->remaining;
        }
        return count;
    }

    /**
     * @brief The `index`-th #BlobMagic::single container (shared-context containers don't count towards
     * `index`).
     * @return An empty (zero-size) view if `index >= blob_count()`.
     */
    HSMContainerView blob_at(size_t index) const noexcept {
        ov::util::MemoryView view = m_view;
        while (view.size() >= sizeof(HSMHeader)) {
            const auto container_view = view;
            const auto next = advance_container(view);
            if (!next) {
                break;
            }
            view = next->remaining;
            if (next->is_blob) {
                if (index == 0) {
                    return HSMContainerView{container_view.data(), container_view.size() - next->remaining.size()};
                }
                --index;
            }
        }
        return HSMContainerView{static_cast<const std::byte*>(nullptr), 0};
    }

private:
    /// Result of walking past one container: remaining bytes after it, and whether it was #BlobMagic::single
    /// (a blob) rather than #BlobMagic::multi (a shared context).
    struct NextContainer {
        ov::util::MemoryView remaining;
        bool is_blob;
    };

    /// `std::nullopt` if `view`'s header is invalid (bad magic, or `total_size` doesn't fit `view`).
    static std::optional<NextContainer> advance_container(const ov::util::MemoryView& view) noexcept {
        const auto& hdr = HSMHeader::view(reinterpret_cast<const uint8_t*>(view.data()));
        if (hdr.magic != BlobMagic::single && hdr.magic != BlobMagic::multi) {
            return std::nullopt;
        }
        if (hdr.version_major != HSMFormatVersion::major) {
            return std::nullopt;
        }
        if (hdr.total_size < sizeof(HSMHeader) || hdr.total_size > view.size()) {
            return std::nullopt;
        }
        const auto container_size = static_cast<size_t>(hdr.total_size);
        return NextContainer{{view.data() + container_size, view.size() - container_size},
                             hdr.magic == BlobMagic::single};
    }

    ov::util::MemoryView m_view;
};

}  // namespace v1

}  // namespace ov::runtime
