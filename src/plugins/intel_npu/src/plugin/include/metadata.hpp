// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/core/version.hpp"
#include "openvino/runtime/tensor.hpp"

namespace intel_npu {

class MetadataBase {
public:
    MetadataBase(uint32_t version, uint64_t blobDataSize);

    using uninitialized_source = void*;
    using Source = std::
        variant<uninitialized_source, std::reference_wrapper<std::istream>, std::reference_wrapper<const ov::Tensor>>;

    /**
     * @brief Reads metadata from a stream.
     */
    void read(std::istream& tensor);

    /**
     * @brief Reads metadata from a ov::Tensor.
     */
    void read(const ov::Tensor& tensor);

    virtual void read() = 0;

    /**
     * @brief Writes metadata to a stream.
     */
    virtual void write(std::ostream& stream) = 0;

    virtual uint64_t get_blob_size() const;

    /**
     * @returns The sizes of the init schedules. Populated only if "weights separation" has been enabled.
     */
    virtual std::optional<std::vector<uint64_t>> get_init_sizes() const;

    virtual std::optional<std::vector<ov::Layout>> get_input_layouts() const;

    virtual std::optional<std::vector<ov::Layout>> get_output_layouts() const;

    /**
     * @returns Batch size. Populated in case of plugin batching.
     */
    virtual std::optional<int64_t> get_batch_size() const;

    virtual ~MetadataBase() = default;

    static std::streampos getFileSize(std::istream& stream);

    virtual size_t get_metadata_size() const = 0;

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

protected:
    /**
     * @brief Reads data from the source containing the metadata. The implementation depends on the type of source.
     */
    void read_data_from_source(char* destination, const size_t size);

    /**
     * @brief Adds the size of the binary object and the magic string to the end of the stream.
     * @details This should be called after the "write" method in order to conclude writing the metadata into the given
     * stream.
     * @note This operation was detached from "write" since "write" writes at the beginning of the stream, while this
     * method writes at the end. This change allows better extension of class hierarchy.
     */
    void append_padding_blob_size_and_magic(std::ostream& stream);

    uint32_t _version;
    uint64_t _blobDataSize;
    Logger _logger;

    /**
     * @brief Where the metadata is read from. The type can be a stream, an OpenVINO tensor or "uninitialized_source".
     * @details Stored as attribute in order to avoid repeatedly passing the same arguments to some methods.
     * "uninitialized_source" (void*) is the default type assigned upon creation.
     */
    Source _source;

    /**
     * @brief Used only when the source buffer is an OV tensor for managing the read coursor.
     */
    size_t _cursorOffset = 0;
};

/**
 * @brief Magic bytes used for identifying NPU blobs.
 */
constexpr std::string_view MAGIC_BYTES = "OVNPU";

/**
 * @brief List of supported version formats.
 */
constexpr uint32_t METADATA_VERSION_2_0{MetadataBase::make_version(2, 0)};
constexpr uint32_t METADATA_VERSION_2_1{MetadataBase::make_version(2, 1)};
constexpr uint32_t METADATA_VERSION_2_2{MetadataBase::make_version(2, 2)};
constexpr uint32_t METADATA_VERSION_2_3{MetadataBase::make_version(2, 3)};

/**
 * @brief Current metadata version.
 */
constexpr uint32_t CURRENT_METADATA_VERSION{METADATA_VERSION_2_3};

constexpr uint16_t CURRENT_METADATA_MAJOR_VERSION{MetadataBase::get_major(CURRENT_METADATA_VERSION)};
constexpr uint16_t CURRENT_METADATA_MINOR_VERSION{MetadataBase::get_minor(CURRENT_METADATA_VERSION)};

class OpenvinoVersion final {
public:
    constexpr OpenvinoVersion(uint16_t major, uint16_t minor, uint16_t patch)
        : _major(major),
          _minor(minor),
          _patch(patch) {}

    OpenvinoVersion(const OpenvinoVersion& version);

    OpenvinoVersion& operator=(const OpenvinoVersion& other) {
        if (this != &other) {
            _major = other.get_major();
            _minor = other.get_minor();
            _patch = other.get_patch();
        }

        return *this;
    }

    ~OpenvinoVersion() = default;

    /**
     * @brief Reads version data from a stream.
     */
    void read(std::istream& istream);

    /**
     * @brief Reads version data from a ov::Tensor.
     */
    void read(const ov::Tensor& tensor);

    /**
     * @brief Writes version data to a stream.
     */
    void write(std::ostream& stream);

    uint16_t get_major() const;

    uint16_t get_minor() const;

    uint16_t get_patch() const;

    size_t get_openvino_version_size() const;

    bool operator!=(const OpenvinoVersion& version);

private:
    uint16_t _major;
    uint16_t _minor;
    uint16_t _patch;
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
 * @brief Template specialization for metadata version 2.0.
 */
template <>
class Metadata<METADATA_VERSION_2_0> : public MetadataBase {
public:
    Metadata(uint64_t blobSize, const std::optional<OpenvinoVersion>& ovVersion = std::nullopt);

    void read() override;

    /**
     * @attention It's a must to first write metadata version in any metadata specialization.
     *
     * @details When importing a versioned blob, it's best to first read the metadata version field.
     * This is the quickest way to handle many incompatible blob cases without needing to traverse the whole NPU
     * metadata section.
     */
    void write(std::ostream& stream) override;

    size_t get_metadata_size() const override;

protected:
    OpenvinoVersion _ovVersion;
};

/**
 * @brief The version that adds support for init schedules (weights separation).
 */
template <>
class Metadata<METADATA_VERSION_2_1> : public Metadata<METADATA_VERSION_2_0> {
public:
    Metadata(uint64_t blobSize,
             const std::optional<OpenvinoVersion>& ovVersion = std::nullopt,
             const std::optional<std::vector<uint64_t>>& initSizes = std::nullopt);

    /**
     * @details The number of init schedules, along with the size of each init binary object are read in addition to the
     * information provided by the previous metadata versions.
     */
    void read() override;

    /**
     * @details The number of init schedules, along with the size of each init binary object are written in addition to
     * the information registered by the previous metadata versions.
     */
    void write(std::ostream& stream) override;

    std::optional<std::vector<uint64_t>> get_init_sizes() const override;

    size_t get_metadata_size() const override;

private:
    std::optional<std::vector<uint64_t>> _initSizes;
    uint64_t _numberOfInits = 0;
};

/**
 * @brief The version that adds support for batch value storage.
 */
template <>
class Metadata<METADATA_VERSION_2_2> : public Metadata<METADATA_VERSION_2_1> {
public:
    Metadata(uint64_t blobSize,
             std::optional<OpenvinoVersion> ovVersion = std::nullopt,
             const std::optional<std::vector<uint64_t>> initSizes = std::nullopt,
             const std::optional<int64_t> batchSize = std::nullopt);

    void read() override;

    void write(std::ostream& stream) override;

    std::optional<int64_t> get_batch_size() const override;

    size_t get_metadata_size() const override;

private:
    std::optional<int64_t> _batchSize;
};

/**
 * @brief Stores the layouts for all inputs and outputs (Parameter and Result nodes).
 * @details The order used for recording the layouts follows the deterministic order in which OV parses the I/O.
 */
template <>
class Metadata<METADATA_VERSION_2_3> : public Metadata<METADATA_VERSION_2_2> {
public:
    Metadata(uint64_t blobSize,
             const std::optional<OpenvinoVersion>& ovVersion = std::nullopt,
             const std::optional<std::vector<uint64_t>>& initSizes = std::nullopt,
             const std::optional<int64_t> batchSize = std::nullopt,
             const std::optional<std::vector<ov::Layout>>& inputLayouts = std::nullopt,
             const std::optional<std::vector<ov::Layout>>& outputLayouts = std::nullopt);

    void read() override;

    void write(std::ostream& stream) override;

    size_t get_metadata_size() const override;

    std::optional<std::vector<ov::Layout>> get_input_layouts() const override;

    std::optional<std::vector<ov::Layout>> get_output_layouts() const override;

private:
    std::optional<std::vector<ov::Layout>> _inputLayouts;
    std::optional<std::vector<ov::Layout>> _outputLayouts;
};

/**
 * @brief Creates a Metadata object.
 *
 * @return Unique pointer to the created MetadataBase object if the major version is supported; otherwise, returns
 * 'nullptr'.
 */
std::unique_ptr<MetadataBase> create_metadata(uint32_t version, uint64_t blobSize);

/**
 * @brief Reads metadata from a blob (istream).
 *
 * @return If the blob is versioned and its major version is supported, returns an unique pointer to the read
 * MetadataBase object; otherwise, returns 'nullptr'.
 *
 * @note The version check can be disabled if the "OV_NPU_DISABLE_VERSION_CHECK" environment variable is set to
 * 'YES'.
 */
std::unique_ptr<MetadataBase> read_metadata_from(std::istream& stream);

/**
 * @brief Reads metadata from a blob (ov::Tensor).
 *
 * @return If the blob is versioned and its major version is supported, returns an unique pointer to the read
 * MetadataBase object; otherwise, returns 'nullptr'.
 */
std::unique_ptr<MetadataBase> read_metadata_from(const ov::Tensor& tensor);

}  // namespace intel_npu
