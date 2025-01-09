// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "metadata.hpp"

#include <cstring>
#include <optional>
#include <sstream>

#include "intel_npu/config/config.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/core/version.hpp"

namespace {
std::streampos getFileSize(std::istream& stream) {
    auto log = intel_npu::Logger::global().clone("getFileSize");
    if (!stream) {
        OPENVINO_THROW("Stream is in bad status! Please check the passed stream status!");
    }

    if (stream.rdbuf()->in_avail() != 0) {
        // ov::OwningSharedStreamBuffer scenario
        return stream.rdbuf()->in_avail() + stream.tellg();
    }
    const std::streampos streamStart = stream.tellg();
    stream.seekg(0, std::ios_base::end);
    const std::streampos streamEnd = stream.tellg();
    stream.seekg(streamStart, std::ios_base::beg);

    log.debug("Read blob size: streamStart=%zu, streamEnd=%zu", streamStart, streamEnd);

    if (streamEnd < streamStart) {
        OPENVINO_THROW("Invalid stream size: streamEnd (",
                       streamEnd,
                       ") is not larger than streamStart (",
                       streamStart,
                       ")!");
    }

    return streamEnd - streamStart;
}
}  // anonymous namespace

namespace intel_npu {

OpenvinoVersion::OpenvinoVersion(std::string_view version)
    : _version(version),
      _size(static_cast<uint32_t>(version.size())) {}

void OpenvinoVersion::read(std::istream& stream) {
    stream.read(reinterpret_cast<char*>(&_size), sizeof(_size));
    _version.resize(_size);
    stream.read(_version.data(), _size);
}

void OpenvinoVersion::write(std::ostream& stream) {
    stream.write(reinterpret_cast<const char*>(&_size), sizeof(_size));
    stream.write(_version.data(), _size);
}

Metadata<METADATA_VERSION_1_0>::Metadata(uint64_t blobSize, std::optional<std::string_view> ovVersion)
    : _version{METADATA_VERSION_1_0},
      _ovVersion{ovVersion.value_or(ov::get_openvino_version().buildNumber)},
      _blobDataSize{blobSize} {}

void Metadata<METADATA_VERSION_1_0>::read(std::istream& stream) {
    _ovVersion.read(stream);
}

void Metadata<METADATA_VERSION_1_0>::write(std::ostream& stream) {
    stream.write(reinterpret_cast<const char*>(&_version), sizeof(_version));
    _ovVersion.write(stream);
    stream.write(reinterpret_cast<const char*>(&_blobDataSize), sizeof(_blobDataSize));
    stream.write(MAGIC_BYTES.data(), MAGIC_BYTES.size());
}

std::unique_ptr<MetadataBase> create_metadata(uint32_t version, uint64_t blobSize) {
    switch (version) {
    case METADATA_VERSION_1_0:
        return std::make_unique<Metadata<METADATA_VERSION_1_0>>(blobSize, std::nullopt);

    default:
        OPENVINO_THROW("Invalid metadata version!");
    }
}

std::string OpenvinoVersion::get_version() const {
    return _version;
}

bool Metadata<METADATA_VERSION_1_0>::is_compatible() {
    auto logger = Logger::global().clone("NPUBlobMetadata");
    // checking if we can import the blob
    if (_ovVersion.get_version() != ov::get_openvino_version().buildNumber) {
        logger.warning("Imported blob OpenVINO version: %s, but the current OpenVINO version is: %s",
                       _ovVersion.get_version().c_str(),
                       ov::get_openvino_version().buildNumber);

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
        if (auto envVar = std::getenv("NPU_DISABLE_VERSION_CHECK")) {
            if (envVarStrToBool("NPU_DISABLE_VERSION_CHECK", envVar)) {
                return true;
            }
        }
#endif
        return false;
    }
    return true;
}

std::unique_ptr<MetadataBase> read_metadata_from(std::istream& stream) {
    auto logger = Logger::global().clone("NPUBlobMetadata");
    size_t magicBytesSize = MAGIC_BYTES.size();
    std::string blobMagicBytes;
    blobMagicBytes.resize(magicBytesSize);

    std::streampos currentStreamPos = stream.tellg(), streamSize = getFileSize(stream);
    stream.seekg(-currentStreamPos + streamSize - magicBytesSize, std::ios::cur);
    stream.read(blobMagicBytes.data(), magicBytesSize);
    if (MAGIC_BYTES != blobMagicBytes) {
        OPENVINO_THROW("Blob is missing NPU metadata!");
    }

    uint64_t blobDataSize;
    stream.seekg(-std::streampos(magicBytesSize) - sizeof(blobDataSize), std::ios::cur);
    stream.read(reinterpret_cast<char*>(&blobDataSize), sizeof(blobDataSize));
    stream.seekg(-stream.tellg() + currentStreamPos + blobDataSize, std::ios::cur);

    uint32_t metaVersion;
    stream.read(reinterpret_cast<char*>(&metaVersion), sizeof(metaVersion));

    std::unique_ptr<MetadataBase> storedMeta;
    try {
        storedMeta = create_metadata(metaVersion, blobDataSize);
        storedMeta->read(stream);
    } catch (...) {
        logger.warning("Imported blob metadata version: %d.%d, but the current version is: %d.%d",
                       get_major(metaVersion),
                       get_minor(metaVersion),
                       get_major(CURRENT_METADATA_VERSION),
                       get_minor(CURRENT_METADATA_VERSION));

        OPENVINO_THROW("NPU metadata mismatch.");
    }
    stream.seekg(-stream.tellg() + currentStreamPos, std::ios::cur);

    return storedMeta;
}

uint64_t Metadata<METADATA_VERSION_1_0>::get_blob_size() const {
    return _blobDataSize;
}

}  // namespace intel_npu
