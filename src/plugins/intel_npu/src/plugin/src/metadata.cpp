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

Metadata<METADATA_VERSION_1_0>::Metadata(std::optional<std::string_view> ovVersion)
    : _version{METADATA_VERSION_1_0},
      _ovVersion{ovVersion.value_or(ov::get_openvino_version().buildNumber)} {}

void Metadata<METADATA_VERSION_1_0>::read(std::istream& stream) {
    _ovVersion.read(stream);
}

void Metadata<METADATA_VERSION_1_0>::write(std::ostream& stream) {
    stream.write(reinterpret_cast<const char*>(&_version), sizeof(_version));
    _ovVersion.write(stream);
}

std::unique_ptr<MetadataBase> create_metadata(uint32_t version) {
    switch (version) {
    case METADATA_VERSION_1_0:
        return std::make_unique<Metadata<METADATA_VERSION_1_0>>(std::nullopt);

    default:
        OPENVINO_THROW("Invalid metadata version!");
    }
}

std::string OpenvinoVersion::get_version() {
    return _version;
}

bool Metadata<METADATA_VERSION_1_0>::is_compatible() {
    Logger logger("NPUPlugin", Logger::global().level());
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

std::unique_ptr<MetadataBase> read_metadata_from(const std::vector<uint8_t>& blob) {
    Logger logger("NPUPlugin", Logger::global().level());
    size_t magicBytesSize = MAGIC_BYTES.size();
    std::string blobMagicBytes;
    blobMagicBytes.resize(magicBytesSize);

    auto metadataIterator = blob.end() - magicBytesSize;
    std::memcpy(blobMagicBytes.data(), &(*metadataIterator), magicBytesSize);
    if (MAGIC_BYTES != blobMagicBytes) {
        OPENVINO_THROW("Blob is missing NPU metadata!");
    }

    uint64_t blobDataSize;
    metadataIterator -= sizeof(blobDataSize);
    std::memcpy(&blobDataSize, &(*metadataIterator), sizeof(blobDataSize));
    metadataIterator = blob.begin() + blobDataSize;

    std::stringstream metadataStream;
    metadataStream.write(reinterpret_cast<const char*>(&(*metadataIterator)),
                         blob.end() - metadataIterator - sizeof(blobDataSize));

    uint32_t metaVersion;
    metadataStream.read(reinterpret_cast<char*>(&metaVersion), sizeof(metaVersion));

    std::unique_ptr<MetadataBase> storedMeta;
    try {
        storedMeta = create_metadata(metaVersion);
        storedMeta->read(metadataStream);
    } catch (...) {
        logger.warning("Imported blob metadata version: %d.%d, but the current version is: %d.%d",
                       get_major(metaVersion),
                       get_minor(metaVersion),
                       get_major(CURRENT_METADATA_VERSION),
                       get_minor(CURRENT_METADATA_VERSION));

        OPENVINO_THROW("NPU metadata mismatch.");
    }
    return storedMeta;
}

}  // namespace intel_npu
