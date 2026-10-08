// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief HSM (Header-Sections-Manifest) blob container format contract.
 * @file openvino/runtime/hsm_format.hpp
 *
 * @verbatim
   +---------------------------+---------------------------+---------------------------+
   |           Header          |      Section payloads     |          Manifest         |
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
#include "openvino/runtime/common.hpp"
#include "openvino/util/container_util.hpp"
#include "openvino/util/memory.hpp"

namespace ov::runtime::hsm {

using MagicType = std::array<uint8_t, 5>;  //!< 5 raw ASCII bytes: container magic at offset 0.
using SizeType = uint64_t;                 //!< Container/section size fields.
using OffsetType = uint64_t;               //!< Byte offsets within the container.

/// Container magic (5 raw bytes).
struct BlobMagic {
    MagicType value{};  //!< Raw bytes. Zero-initialized: an unset BlobMagic never matches a valid magic.

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
    static const BlobMagic multi;   //!< "OVWSH": shared-context container - see #MultiBlobView.
};

inline constexpr BlobMagic BlobMagic::single{ov::util::make_array<uint8_t>('O', 'V', 'B', 'L', 'S')};
inline constexpr BlobMagic BlobMagic::multi{ov::util::make_array<uint8_t>('O', 'V', 'W', 'S', 'H')};

/**
 * @brief The currently-active HSM v1.x format, as an `inline namespace` nested under #ov::runtime::hsm:
 * `Header`, `FormatVersion`, `SectionTag`, `Tags`, `ManifestEntry` and friends all resolve as
 * `ov::runtime::hsm::Header` etc. A future major version is a plain `namespace v2` alongside this one,
 * explicit until promoted by moving `inline` here. Only #BlobMagic and the basic type aliases above stay
 * shared across every version.
 */
inline namespace v1 {

/**
 * @brief HSM format version written by this codebase.
 * @note Compatibility rule: a reader requires `header.version_major == major` exactly. #minor is additive-only
 * (new optional tags/fields never gate compatibility) - a reader must accept any #minor value once #major matches,
 * and must never fail solely on a #minor mismatch.
 */
struct FormatVersion {
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
   | 8      | 8    | container_size  |
   | 16     | 8    | manifest_offset |
   | 24     | 8    | manifest_size   |
   +--------+------+-----------------+
   @endverbatim
 *
 * @note Native byte order (safe: all supported targets are little-endian); `#pragma pack(1)` keeps fields at
 * their documented offsets with no implicit padding.
 */
#pragma pack(push, 1)
struct Header {
    BlobMagic magic;         //!< Container magic - see #BlobMagic.
    uint16_t version_major;  //!< Format major version this container was written with - see #FormatVersion.
    uint8_t version_minor;   //!< Format minor version this container was written with - see #FormatVersion.

    SizeType container_size;     //!< Whole container size, in bytes.
    OffsetType manifest_offset;  //!< Byte offset of the first ManifestEntry.
    SizeType manifest_size;      //!< Manifest size in bytes; entry count = manifest_size / sizeof(ManifestEntry).

    /**
     * @brief Non-owning view of the header at the start of an HSM container buffer - reinterprets the bytes
     * in place, no copy.
     * @warning No bounds checking: caller must ensure `data` points to at least `sizeof(Header)` readable bytes.
     */
    static const Header& view(const uint8_t* data) noexcept {
        return *reinterpret_cast<const Header*>(data);
    }
};
#pragma pack(pop)
static_assert(sizeof(Header) == 32,
              "Header layout changed - bump FormatVersion::major/document the change before touching "
              "this struct; readers and writers must be updated together.");

/**
 * @brief Device/plugin that owns a ManifestEntry's section - see #Devices for the catalog of assigned ids.
 * @note Paired with #SectionTag: a reader always compares `(device, tag)` together, never `tag` alone, so two
 * devices may reuse the same tag id for unrelated content. Conversely, a device needing content different from
 * a Core tag's shared meaning writes that same tag id under its own id rather than inventing a new tag; a
 * reader resolving such a tag looks under its own device first and falls back to #any_device_id.
 */
using DeviceId = uint8_t;

/**
 * @brief Catalog of assigned #DeviceId values; never change or reuse a value once shipped.
 * #sentinel_count auto-tracks the count and must stay last.
 * @note Meta-plugins (Hetero, AUTO, AUTO-BATCH) deliberately have no id: they redirect to real devices rather
 * than compute themselves, so the device owning a section is always the one that produced it.
 */
enum class Devices : DeviceId {
    any = 0,  //!< See #any_device_id.
    cpu = 1,  //!< See #cpu_device_id.
    gpu = 2,  //!< See #gpu_device_id.
    npu = 3,  //!< See #npu_device_id.
    // Add new real devices above this line only, each with the next explicit value - never change or reuse an
    // existing device's value.
    sentinel_count,  // Not a real device id - always exactly one past the last real entry above.
};

/**
 * @brief First #DeviceId of the sample/test bucket, kept clear of the #Devices range so a sample plugin can
 * take an id without ever colliding with a real device.
 */
inline constexpr DeviceId sample_device_id_range_start = 100;
static_assert(static_cast<DeviceId>(Devices::sentinel_count) <= sample_device_id_range_start,
              "Too many real devices defined for sample_device_id_range_start - widen the boundary.");

/** @brief Reserved #DeviceId for sections not tied to one specific device. */
inline constexpr DeviceId any_device_id = static_cast<DeviceId>(Devices::any);

inline constexpr DeviceId cpu_device_id = static_cast<DeviceId>(Devices::cpu);  //!< CPU plugin.
inline constexpr DeviceId gpu_device_id = static_cast<DeviceId>(Devices::gpu);  //!< GPU plugin.
inline constexpr DeviceId npu_device_id = static_cast<DeviceId>(Devices::npu);  //!< NPU plugin.

/**
 * @brief Boundary in the 23-bit #SectionTag id space: Core ids are `< core_tag_id_range_end`;
 * #SectionTag::make_device_tag() only produces ids at/above it, so a device can never collide with a Core tag.
 */
inline constexpr uint32_t core_tag_id_range_end = 0x1000;

/** @brief Maximum tag id representable by the 23-bit #SectionTag id space. */
inline constexpr uint32_t max_tag_id = 0x7FFFFF;

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

    /**
     * @brief Builds a device-specific wire SectionTag from a device-local id; always lands at/above
     * #core_tag_id_range_end, so it can never collide with a Core tag.
     * @param local_id Device-local id, starting at 0; must be `<= max_tag_id - core_tag_id_range_end`.
     * @param is_inline true for inline-mode, false for pointer-mode.
     */
    static constexpr SectionTag make_device_tag(uint32_t local_id, bool is_inline) noexcept {
        OPENVINO_DEBUG_ASSERT(local_id <= max_tag_id - core_tag_id_range_end);
        return make(core_tag_id_range_end + local_id, is_inline);
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

    /// Inverse of #make_device_tag(): recovers the device-local id (`id() - core_tag_id_range_end`).
    constexpr uint32_t device_local_id() const noexcept {
        return id() - core_tag_id_range_end;
    }
};

/**
 * @brief Core-owned HSM tag identifiers - explicit wire values so reordering is safe; never change or reuse a
 * value once shipped. #sentinel_count auto-tracks the count and must stay last.
 */
enum class Tags : uint32_t {
    invalid = 0,               //!< Reserved: never a real tag id.
    model_id = 1,              //!< See #model_id.
    model = 2,                 //!< See #model.
    runtime_requirements = 3,  //!< See #runtime_requirements_tag.
    model_struct = 4,          //!< See #model_struct_tag.
    weights = 5,               //!< See #weights_tag.
    compiled_options = 6,      //!< See #compiled_options_tag.
    // Add new Core tags above this line only, each with the next explicit value - never change or reuse an
    // existing tag's value.
    sentinel_count,  // Not a real tag id - always exactly one past the last real entry above.
};
static_assert(static_cast<uint32_t>(Tags::sentinel_count) <= core_tag_id_range_end,
              "Too many Core tags defined for core_tag_id_range_end - widen the boundary.");

inline constexpr uint32_t model_id = static_cast<uint32_t>(Tags::model_id);  //!< Model identifier (e.g. a hash).
inline constexpr uint32_t model = static_cast<uint32_t>(Tags::model);        //!< The serialized compiled model itself.
inline constexpr uint32_t runtime_requirements = static_cast<uint32_t>(Tags::runtime_requirements);
inline constexpr uint32_t model_struct = static_cast<uint32_t>(Tags::model_struct);
inline constexpr uint32_t weights = static_cast<uint32_t>(Tags::weights);
inline constexpr uint32_t compiled_options = static_cast<uint32_t>(Tags::compiled_options);

/// Wire tag for #model_id - always inline-mode.
inline constexpr SectionTag model_id_tag = SectionTag::make(model_id, /*is_inline=*/true);

/// Wire tag for #model - always pointer-mode.
inline constexpr SectionTag model_tag = SectionTag::make(model, /*is_inline=*/false);

/**
 * @brief Wire tag for #runtime_requirements - always pointer-mode. Payload is opaque to the common
 * reader/format: this contract only reserves the tag and its bounds (like any pointer-mode section) -
 * interpreting and enforcing the encoded requirements is entirely the emitting device/plugin's
 * responsibility, typically via #ISectionExtension. No expression scheme is defined at this layer
 * (out of scope here; a richer format, if any, belongs to the tag registry).
 */
inline constexpr SectionTag runtime_requirements_tag = SectionTag::make(runtime_requirements, /*is_inline=*/false);

/// Wire tag for #model_struct - always pointer-mode. Payload contract: CVS-196450.
inline constexpr SectionTag model_struct_tag = SectionTag::make(model_struct, /*is_inline=*/false);

/// Wire tag for #weights - always pointer-mode. Payload contract: CVS-196451.
inline constexpr SectionTag weights_tag = SectionTag::make(weights, /*is_inline=*/false);

/// Wire tag for #compiled_options - always pointer-mode. Payload contract: CVS-196452.
inline constexpr SectionTag compiled_options_tag = SectionTag::make(compiled_options, /*is_inline=*/false);

/**
 * @brief One fixed-size, 32-byte record of the manifest table (see #Header::manifest_offset).
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
 * @note `tag_reserved`/`pointer_reserved` are zero unless the specific `(device, tag)` pair they belong to
 * redefines their meaning - treat them as opaque otherwise.
 */
#pragma pack(push, 1)
struct ManifestEntry {
    DeviceId device;                      //!< Owning device.
    SectionTag tag;                       //!< Section content;
    std::array<uint8_t, 4> tag_reserved;  //!< Zero, unless the specific (#device, #tag) pair redefines this.

    union {
        struct {
            OffsetType offset;                        //!< Section payload offset (pointer-mode).
            SizeType size;                            //!< Section payload size (pointer-mode).
            std::array<uint8_t, 8> pointer_reserved;  //!< Zero, unless (#device, #tag) redefines this.
        };
        std::array<uint8_t, 24> inline_bytes;  //!< Inline payload, up to 24 bytes (inline-mode).
    };
};
#pragma pack(pop)
static_assert(sizeof(ManifestEntry) == 32, "ManifestEntry layout changed.");

/**
 * @brief Checks if the HSM header has a recognized magic number and version.
 *
 * @param header The HSM header to check.
 * @return true if the header has a recognized magic number and version, false otherwise.
 */
constexpr bool is_recognized_header(const Header& header) noexcept {
    return (header.magic == BlobMagic::single || header.magic == BlobMagic::multi) &&
           header.version_major == FormatVersion::major && header.container_size >= sizeof(Header);
}

/**
 * @brief Checks that the header fields satisfy basic consistency rules.
 *
 * Ensures that the magic number is valid, the version matches the expected format, the total size is at least
 * as large as the header, the manifest size is a multiple of the manifest entry size, and the manifest offset
 * and size fit within the total size.
 * @param header The HSM header to validate.
 * @return true if the header fields satisfy the basic consistency rules, false otherwise.
 */
constexpr bool is_valid_header_fields(const Header& header) noexcept {
    return is_recognized_header(header) && header.manifest_size % sizeof(ManifestEntry) == 0 &&
           header.manifest_offset >= sizeof(Header) && header.manifest_offset <= header.container_size &&
           header.container_size - header.manifest_offset >= header.manifest_size;
}

/**
 * @brief Checks that a manifest entry's section bounds are valid within the container.
 *
 * Returns true for an inline-mode entry (nothing to bounds-check), or a pointer-mode one whose offset/size stay
 * within `header.manifest_offset` (payload always precedes the manifest - see #Header::manifest_offset).
 * @param entry The manifest entry to validate.
 * @param header The HSM header providing context for bounds checking.
 * @return true if the section bounds are valid, false otherwise.
 */
constexpr bool is_valid_section_bounds(const ManifestEntry& entry, const Header& header) noexcept {
    return entry.tag.is_inline() || (entry.offset >= sizeof(Header) && entry.offset <= header.manifest_offset &&
                                     header.manifest_offset - entry.offset >= entry.size);
}

/**
 * @brief Reader-side plugin hook: interprets one manifest entry's section content. A concrete extension
 * self-dispatches by checking `(entry.device, entry.tag)` and returning whether it recognized it - per the
 * unknown-tag rule on #SectionTag, the reader must skip any entry no extension recognizes, never fail
 * import.
 * @note Forward-looking contract only - not yet wired to a real reader.
 */
class ISectionExtension {
public:
    virtual ~ISectionExtension() = default;

    /**
     * @brief Attempts to interpret @p entry's section content.
     * @param entry Manifest entry being considered - not necessarily one this extension owns.
     * @param section Bounds-checked view of the payload - #ContainerView::section() for a pointer-mode
     * entry, or `entry.inline_bytes` for an inline-mode one; never a raw, unchecked pointer.
     * @return true if `(entry.device, entry.tag)` was recognized and handled, false otherwise.
     */
    virtual bool read_section(const ManifestEntry& entry, ov::util::MemoryView section) = 0;
};

/**
 * @brief Read-only, zero-copy view of an entire in-memory HSM container: header, manifest and pointer-mode
 */
class OPENVINO_RUNTIME_API ContainerView {
public:
    /// Empty (zero-size, null-data) view - #validate() is false for it.
    constexpr ContainerView() noexcept = default;
    explicit constexpr ContainerView(const std::byte* data, size_t size) noexcept : m_view{data, size} {}
    explicit ContainerView(const uint8_t* data, size_t size) noexcept
        : ContainerView{reinterpret_cast<const std::byte*>(data), size} {}

    constexpr size_t size() const noexcept {
        return m_view.size();
    }

    /**
     * @brief Returns the header at the start of the buffer.
     * @return Reference to the header at the start of the buffer.
     */
    const Header& header() const noexcept;

    /**
     * @brief Returns the first manifest entry at `header().manifest_offset`.
     * @return The first manifest entry at `header().manifest_offset`.
     */
    const ManifestEntry& manifest() const noexcept;

    /// Number of entries at #manifest().
    size_t manifest_count() const noexcept {
        return static_cast<size_t>(header().manifest_size / sizeof(ManifestEntry));
    }

    /**
     * @brief Bounds-checked payload bytes of a pointer-mode manifest entry; empty view for an invalid or inline entry.
     */
    constexpr ov::util::MemoryView section(const ManifestEntry& entry) const noexcept {
        if (entry.tag.is_inline() || entry.offset > size() || entry.size > size() - entry.offset) {
            return {};
        } else {
            return {begin() + static_cast<size_t>(entry.offset), static_cast<size_t>(entry.size)};
        }
    }

    /**
     * @brief Basic structural integrity check: magic, and that the header/manifest/pointer-mode section
     * bounds all stay within #size() with no overflow. Doesn't interpret tag-specific (device, tag) content.
     */
    bool validate() const noexcept;

private:
    constexpr const std::byte* begin() const noexcept {
        return m_view.begin();
    }
    constexpr const std::byte* end() const noexcept {
        return m_view.end();
    }

    ov::util::MemoryView m_view{};
};

/**
 * @brief View over a multi-blob HSM file, providing access to individual blob containers.
 *
 * This class allows iterating over and accessing the #BlobMagic::single containers within a multi-blob HSM file,
 * skipping over shared-context containers.

 */
class OPENVINO_RUNTIME_API MultiBlobView {
public:
    // CVS-191965: finalize this class (multi-blob/shared-context implementation).
    /// Constructs a view over `[data, data + size)` - no copy, no validation of its contents.
    explicit constexpr MultiBlobView(const std::byte* data, size_t size) noexcept : m_view{data, size} {}

    /// @overload uint8_t variant of MultiBlobView(const std::byte*, size_t).
    explicit MultiBlobView(const uint8_t* data, size_t size) noexcept
        : MultiBlobView{reinterpret_cast<const std::byte*>(data), size} {}

    constexpr size_t size() const noexcept {
        return m_view.size();
    }

    /**
     * @brief Returns the number of #BlobMagic::single containers in the multi-blob HSM file.
     * @return The number of #BlobMagic::single containers in the multi-blob HSM file.
     */
    size_t blob_count() const noexcept;

    /**
     * @brief The `index`-th #BlobMagic::single container (shared-context containers don't count towards `index`).
     * @return An empty (zero-size) view if `index >= blob_count()`.
     */
    ContainerView blob_at(size_t index) const noexcept;

private:
    /**
     * @brief Describes the result of advancing past one container in a memory view.
     *
     * This struct contains the remaining view after the container and a flag indicating whether the container was a
     * blob.
     */
    struct NextContainer {
        // Definitions in hsm_format.cpp: only used there, so out-of-line costs no inlining (same TU).
        NextContainer(ov::util::MemoryView remaining, size_t container_size, bool is_blob) noexcept;

        const ov::util::MemoryView& remaining() const noexcept;
        size_t container_size() const noexcept;
        bool is_blob() const noexcept;

    private:
        ov::util::MemoryView m_remaining;  //!< The remaining view after the container.
        size_t m_container_size;           //!< Size of the container just advanced past.
        bool m_is_blob;                    //!< True if the container was a blob.
    };

    /**
     * @brief Advances past one container in the given view.
     * @param view The memory view starting at the container header.
     * @return A NextContainer describing the remaining view and whether it was a blob, or `std::nullopt` if the header
     * is invalid.
     */
    static std::optional<NextContainer> advance_container(const ov::util::MemoryView& view) noexcept;

    ov::util::MemoryView m_view;
};

}  // namespace v1
}  // namespace ov::runtime::hsm
