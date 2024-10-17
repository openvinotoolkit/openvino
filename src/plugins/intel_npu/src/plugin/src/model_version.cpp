// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_version.hpp"

namespace intel_npu {

OpenvinoVersion::OpenvinoVersion(const std::string& version) {
    this->version = version;
    this->size = version.size();
}

void OpenvinoVersion::read(std::istream& stream) {
    stream.read(reinterpret_cast<char*>(&size), sizeof(size));
    version.resize(size);
    stream.read(&version[0], size);
}

Metadata<1, 0>::Metadata() : version{1, 0}, ovVersion{ov::get_openvino_version().buildNumber} {}

std::stringstream Metadata<1, 0>::data() {
    std::stringstream stream;

    stream.write(VERSION_HEADER.data(), VERSION_HEADER.size());

    stream.write(reinterpret_cast<const char*>(&version.major), sizeof(version.major));

    stream.write(reinterpret_cast<const char*>(&version.minor), sizeof(version.minor));

    stream.write(ovVersion.version.c_str(), ovVersion.version.size());

    return stream;
}

void Metadata<1, 0>::write(std::ostream& stream) {
    std::stringstream metav1_data = data();

    stream << metav1_data.rdbuf();
}

void Metadata<1, 0>::read(std::istream& stream) {
    ovVersion.read(stream);
}

void check_blob_version(std::vector<uint8_t>& blob, std::istream& stream) {
    size_t blobDataSize;
    auto metadataIterator = blob.end() - sizeof(size_t);
    memcpy(&blobDataSize, &(*metadataIterator), sizeof(blobDataSize));
    if (blobDataSize == blob.size() - sizeof(blobDataSize)) {
        OPENVINO_THROW("Imported blob is not versioned");
    }

    metadataIterator = blob.begin() + blobDataSize;

    char* blobVersionHeader = new char[VERSION_HEADER.size() + 1];
    blobVersionHeader[VERSION_HEADER.size()] = '\0';
    std::copy(metadataIterator, metadataIterator + VERSION_HEADER.size(), blobVersionHeader);
    std::cout << "header: " << blobVersionHeader << '\n';

    if (VERSION_HEADER != std::string_view(blobVersionHeader, VERSION_HEADER.size())) {
        delete[] blobVersionHeader;
        OPENVINO_THROW("Version header mismatch or missing");
    }
    metadataIterator += VERSION_HEADER.size();

    MetadataVersion metaVersion;
    memcpy(&metaVersion.major, &(*metadataIterator), sizeof(metaVersion.major));
    metadataIterator += sizeof(uint32_t);
    std::cout << "major: " << metaVersion.major;

    memcpy(&metaVersion.minor, &(*metadataIterator), sizeof(metaVersion.minor));
    std::cout << "\nminor: " << metaVersion.minor << '\n';
    metadataIterator += sizeof(uint32_t);

    auto meta = Metadata<CURRENT_METAVERSION_MAJOR, CURRENT_METAVERSION_MINOR>();
    meta.read(stream);

    delete[] blobVersionHeader;
}

} // namespace intel_npu