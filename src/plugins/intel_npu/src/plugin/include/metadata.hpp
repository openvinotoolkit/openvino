// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "openvino/core/version.hpp"

namespace intel_npu {

struct MetadataBase {
protected:
    uint32_t _version;

public:
    MetadataBase(uint32_t version) : _version(version) {}

    /**
     * @brief Reads metadata from a stream.
     */
    virtual void read(std::istream& stream) = 0;

    /**
     * @brief Writes metadata to a stream.
     */
    virtual void write(std::ostream& stream) = 0;

    virtual bool is_compatible() = 0;

    virtual uint64_t get_blob_size() const = 0;

    virtual ~MetadataBase() = default;

    static std::streampos getFileSize(std::istream& stream);

    /**
     * @brief Returns a uint32_t value which represents two uint16_t values concatenated.
     * @details Convention for bumping the metadata version:
     *              - Increment Major in case of: removing a current field OR adding a new field in between fields.
     *              - Increment Minor in case of: adding a new field at the end.
     *
     * @return Major and minor versions concatenated into a single uint32_t value.
     */
    static constexpr uint32_t make_version(uint16_t major, uint16_t minor) {
        return major << 16 | (minor & 0x0000ffff);
    }

    /**
     * @brief Gets the major version.
     * @return Major version.
     */
    static constexpr uint16_t get_major(uint32_t version) {
        return static_cast<uint16_t>(version >> 16);
    }

    /**
     * @brief Gets the minor version.
     * @return Minor version.
     */
    static constexpr uint16_t get_minor(uint32_t version) {
        return static_cast<uint16_t>(version);
    }
};

/**
 * @brief Magic bytes used for identifying NPU blobs.
 */
constexpr std::string_view MAGIC_BYTES = "OVNPU";

/**
 * @brief List of supported version formats.
 */
constexpr uint32_t METADATA_VERSION_2_0{MetadataBase::make_version(2, 0)};

/**
 * @brief Current metadata version.
 */
constexpr uint32_t CURRENT_METADATA_VERSION{METADATA_VERSION_2_0};

constexpr uint16_t CURRENT_METADATA_MAJOR_VERSION{MetadataBase::get_major(CURRENT_METADATA_VERSION)};
constexpr uint16_t CURRENT_METADATA_MINOR_VERSION{MetadataBase::get_minor(CURRENT_METADATA_VERSION)};

struct OpenvinoVersion {
private:
    uint16_t _major;
    uint16_t _minor;
    uint16_t _patch;

public:
    constexpr OpenvinoVersion(uint16_t major, uint16_t minor, uint16_t patch)
        : _major(major),
          _minor(minor),
          _patch(patch) {}

    OpenvinoVersion(const OpenvinoVersion& version);

    /**
     * @brief Reads version data from a stream.
     */
    void read(std::istream& stream);

    /**
     * @brief Writes version data to a stream.
     */
    void write(std::ostream& stream);

    uint16_t get_major() const;

    uint16_t get_minor() const;

    uint16_t get_patch() const;

    bool operator!=(const OpenvinoVersion& version);
};

constexpr OpenvinoVersion CURRENT_OPENVINO_VERSION(OPENVINO_VERSION_MAJOR,
                                                   OPENVINO_VERSION_MINOR,
                                                   OPENVINO_VERSION_PATCH);

/**
 * @brief Template for metadata class handling.
 */
template <uint32_t version>
struct Metadata : public MetadataBase {};

/**
 * @brief Template specialization for metadata version 1.0.
 */
template <>
struct Metadata<METADATA_VERSION_2_0> : public MetadataBase {
protected:
    OpenvinoVersion _ovVersion;
    uint64_t _blobDataSize;

public:
    Metadata(uint64_t blobSize, std::optional<OpenvinoVersion> ovVersion = std::nullopt);

    void read(std::istream& stream) override;

    /**
     * @attention It's a must to first write metadata version in any metadata specialization.
     *
     * @details When importing a versioned blob, it's best to first read the metadata version field.
     * This is the quickest way to handle many incompatible blob cases without needing to traverse the whole NPU
     * metadata section.
     */
    void write(std::ostream& stream) override;

    /**
     * @brief Checks if metadata is supported.
     *
     * @return Returns:
     *              - false:
     *                  - if blob metadata does not match current metadata.
     *                  - if blob OpenVINO version does not match current one.
     *
     *              - true: if all versions match.
     *
     * @note The version check can be disabled if the "OV_NPU_DISABLE_VERSION_CHECK" environment variable is set to '1'.
     */
    bool is_compatible() override;

    uint64_t get_blob_size() const override;
};

/**
 * @brief Creates a Metadata object.
 *
 * @return Unique pointer to the created MetadataBase object if the major version is supported; otherwise, returns
 * 'nullptr'.
 */
std::unique_ptr<MetadataBase> create_metadata(uint32_t version, uint64_t blobSize);

/**
 * @brief Reads metadata from a blob.
 *
 * @return If the blob is versioned and its major version is supported, returns an unique pointer to the read
 * MetadataBase object; otherwise, returns 'nullptr'.
 */
std::unique_ptr<MetadataBase> read_metadata_from(std::istream& stream);

}  // namespace intel_npu
