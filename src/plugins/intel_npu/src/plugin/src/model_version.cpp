// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_version.hpp"

namespace intel_npu {

std::vector<uint8_t> Metadata_v1::data() {
    std::vector<uint8_t> metadata;
    std::cout << "v1_data()\n";

    metadata.insert(metadata.end(), reinterpret_cast<uint8_t*>(&this->version.major),
    reinterpret_cast<uint8_t*>(&this->version.major) + sizeof(this->version.major));

    metadata.insert(metadata.end(), reinterpret_cast<uint8_t*>(&this->version.minor),
    reinterpret_cast<uint8_t*>(&this->version.minor) + sizeof(this->version.minor));

    metadata.insert(metadata.end(), this->ovVersion.version.begin(), this->ovVersion.version.end());

    return metadata;
}

// actually what should it return?
void check_blob_version(std::vector<uint8_t>& blob, std::istream& stream) {
    constexpr std::string_view versionHeader{"OVNPU"}; // maybe put this some place else

    size_t blobDataSize; // blob size *WITHOUT* metadata part
    auto metadataIterator {blob.end() - sizeof(size_t)};
    memcpy(&blobDataSize, &(*metadataIterator), sizeof(blobDataSize));
    if (blobDataSize == blob.size() - sizeof(blobDataSize)) {
        OPENVINO_THROW("Imported blob is not versioned");
    }

    metadataIterator = blob.begin() + blobDataSize;

    char* blobVersionHeader = new char[versionHeader.size() + 1];
    std::copy(metadataIterator, metadataIterator + versionHeader.size(), blobVersionHeader);
    std::cout << "header: " << blobVersionHeader << '\n';

    // should we consider the header name changes?
    // if so, we might need multiple header #defines
    if (versionHeader.compare(blobVersionHeader)) {
        std::cout << "expected header: " << versionHeader << "\n actual header: " << blobVersionHeader << '\n';
        OPENVINO_THROW("Version header mismatch or missing");
    }
    metadataIterator += versionHeader.size();

    MetadataVersion metaVersion;
    memcpy(&metaVersion.major, &(*metadataIterator), sizeof(metaVersion.major));
    metadataIterator += sizeof(uint32_t);
    std::cout << "major: " << metaVersion.major;

    memcpy(&metaVersion.minor, &(*metadataIterator), sizeof(metaVersion.minor));
    std::cout << "\nminor: " << metaVersion.minor << '\n';
    metadataIterator += sizeof(uint32_t);

    std::vector<uint8_t> metaVec(metadataIterator, blob.end() - sizeof(size_t));
    std::vector<uint8_t>::iterator metaIt = metaVec.begin();
    // move this to another function?
    if (metaVersion.major == 1) {
        if (metaVersion.minor > 1 && metaVersion.minor < 5) {
            std::cout << "We got Metadata_v1\n";
            Metadata_v1 metav1 {metaVersion};
            metav1.version = metaVersion;
            metav1.read_metadata(metaIt);
        } else if (metaVersion.minor > 6) {
            std::cout << "We got Metadata_v2\n";
            Metadata_v2 metav2 { {metaVersion} };
            metav2.version = metaVersion;
            metav2.read_metadata(metaIt);
        }
    } else if (metaVersion.major == 2) {
        if (metaVersion.minor > 0) {
            std::cout << "We got Metadata_v3\n";
            Metadata_v3 metav3;
            metav3.version = metaVersion;
            metav3.read_metadata(metaIt);
        }
    }
}

void Metadata_v1::read_metadata(std::vector<uint8_t>::iterator& metadataIterator) {
    /*
        is there a way needed to assert the version?
        or do we still want to check for it?
        after all, we can orchestrate everything using metadata major;minor
    */
    std::cout << "inside read: " << this->version.major << "." << this->version.minor << '\n';
    ov::Version ourOvVersion = ov::get_openvino_version();
    size_t ovVersionSize = strlen(ourOvVersion.buildNumber);
    char* blobOvVersion = new char[ovVersionSize + 1];
    std::copy(metadataIterator, metadataIterator + ovVersionSize, blobOvVersion);
    std::cout << "blob ov version: " << blobOvVersion << '\n';
    metadataIterator += ovVersionSize;
}

void Metadata_v1::write_metadata(std::ostream& stream) {
    std::cout << "v1_write()\n";
    const auto metav1_data = this->data();

    stream.write(reinterpret_cast<const char*>(metav1_data.data()), metav1_data.size());
}

std::vector<uint8_t> Metadata_v2::data() {
    std::cout << "v2_data()\n";
    auto metadata {Metadata_v1::data()};

    metadata.insert(metadata.end(), reinterpret_cast<uint8_t*>(&this->layout.something),
    reinterpret_cast<uint8_t*>(&this->layout.something) + sizeof(this->layout.something));

    metadata.insert(metadata.end(), reinterpret_cast<uint8_t*>(&this->layout.somethingElse),
    reinterpret_cast<uint8_t*>(&this->layout.somethingElse) + sizeof(this->layout.somethingElse));

    return metadata;
}

void Metadata_v2::read_metadata(std::vector<uint8_t>::iterator& metadataIterator) {
    std::cout << "v2_read\n";
    std::cout << "inside read: " << this->version.major << "." << this->version.minor << '\n';
    Metadata_v1::read_metadata(metadataIterator);

    memcpy(&this->layout.something, &(*metadataIterator), sizeof(this->layout.something));
    std::cout << "something: " << this->layout.something << '\n';
    metadataIterator += sizeof(int);

    memcpy(&this->layout.somethingElse, &(*metadataIterator), sizeof(this->layout.somethingElse));
    std::cout << "somethingElse: " << this->layout.somethingElse << '\n';
    metadataIterator += sizeof(double);
}

void Metadata_v2::write_metadata(std::ostream& stream) {
    std::cout << "v2_write()\n";
    const auto metav2_data = this->data();
    stream.write(reinterpret_cast<const char*>(metav2_data.data()), metav2_data.size());
}

std::vector<uint8_t> Metadata_v3::data() {
    std::vector<uint8_t> metadata;
    std::cout << "v1_data()\n";

    metadata.insert(metadata.end(), reinterpret_cast<uint8_t*>(&this->version.major),
    reinterpret_cast<uint8_t*>(&this->version.major) + sizeof(this->version.major));

    metadata.insert(metadata.end(), reinterpret_cast<uint8_t*>(&this->version.minor),
    reinterpret_cast<uint8_t*>(&this->version.minor) + sizeof(this->version.minor));

    metadata.insert(metadata.end(), reinterpret_cast<uint8_t*>(&this->layout.something),
    reinterpret_cast<uint8_t*>(&this->layout.something) + sizeof(this->layout.something));

    metadata.insert(metadata.end(), reinterpret_cast<uint8_t*>(&this->layout.somethingElse),
    reinterpret_cast<uint8_t*>(&this->layout.somethingElse) + sizeof(this->layout.somethingElse));

    metadata.insert(metadata.end(), this->ovVersion.version.begin(), this->ovVersion.version.end());

    metadata.insert(metadata.end(), reinterpret_cast<uint8_t*>(&this->extra),
    reinterpret_cast<uint8_t*>(&this->extra) + sizeof(this->extra));

    return metadata;
}

void Metadata_v3::read_metadata(std::vector<uint8_t>::iterator metadataIterator) {
    std::cout << "inside read: " << this->version.major << "." << this->version.minor << '\n';

    memcpy(&this->layout.something, &(*metadataIterator), sizeof(this->layout.something));
    metadataIterator += sizeof(this->layout.something);
    std::cout << "layout.something: " << this->layout.something << '\n';
    memcpy(&this->layout.somethingElse, &(*metadataIterator), sizeof(this->layout.somethingElse));
    metadataIterator += sizeof(this->layout.somethingElse);
    std::cout << "layout.somethingElse: " << this->layout.somethingElse << '\n';

    ov::Version ourOvVersion = ov::get_openvino_version();
    size_t ovVersionSize = strlen(ourOvVersion.buildNumber);
    char* blobOvVersion = new char[ovVersionSize + 1];
    std::copy(metadataIterator, metadataIterator + ovVersionSize, blobOvVersion);
    std::cout << "blob ov version: " << blobOvVersion << '\n';
    metadataIterator += ovVersionSize;

    memcpy(&this->extra, &(*metadataIterator), sizeof(this->extra));
    metadataIterator += sizeof(this->extra);
    std::cout << "extra: " << this->extra << '\n';
}

void Metadata_v3::write_metadata(std::ostream& stream) {
    const auto metav3_data = this->data();
    stream.write(reinterpret_cast<const char*>(metav3_data.data()), metav3_data.size());
}

}