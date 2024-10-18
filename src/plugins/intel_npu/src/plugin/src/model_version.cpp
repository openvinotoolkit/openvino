// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_version.hpp"
#include "intel_npu/utils/logger/logger.hpp"

namespace intel_npu {

OpenvinoVersion::OpenvinoVersion(const std::string& version) {
    this->version = version;
    this->size = static_cast<uint32_t>(version.size());
}

void OpenvinoVersion::read(std::istream& stream) {
    // compare here ov version?
    stream.read(reinterpret_cast<char*>(&size), sizeof(size));
    stream.read(&version[0], size);
}

Metadata<1, 0>::Metadata() : version{1, 0}, ovVersion{ov::get_openvino_version().buildNumber} {}

std::stringstream Metadata<1, 0>::data() const {
    std::stringstream stream;

    stream.write(DELIMITER.data(), DELIMITER.size());

    stream.write(reinterpret_cast<const char*>(&version.major), sizeof(version.major));
    stream.write(reinterpret_cast<const char*>(&version.minor), sizeof(version.minor));

    stream.write(ovVersion.version.c_str(), ovVersion.version.size());

    return stream;
}

void Metadata<1, 0>::write(std::ostream& stream) {
    std::stringstream metaData = data(); // maybe we find a better name
    stream << metaData.rdbuf();
}

void Metadata<1, 0>::read(std::istream& stream) {
    ovVersion.read(stream);
}

void check_blob_version(std::vector<uint8_t>& blob) {
    size_t blobDataSize;
    auto metadataIterator = blob.end() - sizeof(blobDataSize);
    memcpy(&blobDataSize, &(*metadataIterator), sizeof(blobDataSize));
    if (blobDataSize >= blob.size() - sizeof(blobDataSize)) {
        OPENVINO_THROW("Imported blob is not versioned");
    }

    metadataIterator = blob.begin() + blobDataSize;
    std::stringstream metadataStream;
    metadataStream.write(reinterpret_cast<const char*>(&(*metadataIterator)), blob.end() - metadataIterator - sizeof(blobDataSize));

    std::string blobVersionHeader(DELIMITER.size(), '\0');
    metadataStream.read(&blobVersionHeader[0], DELIMITER.size());
    if (DELIMITER != blobVersionHeader) {
        OPENVINO_THROW("Version header mismatch or missing");
    }

    MetadataVersion metaVersion;
    metadataStream.read(reinterpret_cast<char*>(&metaVersion.major), sizeof(metaVersion.major));
    std::cout << "major: " << metaVersion.major;

    metadataStream.read(reinterpret_cast<char*>(&metaVersion.minor), sizeof(metaVersion.minor));
    std::cout << "\nminor: " << metaVersion.minor << '\n';

    auto meta = Metadata<CURRENT_METAVERSION_MAJOR, CURRENT_METAVERSION_MINOR>();
    meta.read(metadataStream);
}

} // namespace intel_npu
