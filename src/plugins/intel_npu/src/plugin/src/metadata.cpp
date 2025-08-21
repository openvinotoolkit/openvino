// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "metadata.hpp"

#include <cstring>
#include <iterator>
#include <optional>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/utils.hpp"
#include "openvino/runtime/shared_buffer.hpp"

namespace intel_npu {

uint16_t OpenvinoVersion::get_major() const {
    return _major;
}

uint16_t OpenvinoVersion::get_minor() const {
    return _minor;
}

uint16_t OpenvinoVersion::get_patch() const {
    return _patch;
}

bool OpenvinoVersion::operator!=(const OpenvinoVersion& version) {
    return this->_major != version._major || this->_minor != version._minor || this->_patch != version._patch;
}

OpenvinoVersion::OpenvinoVersion(const OpenvinoVersion& version)
    : _major(version.get_major()),
      _minor(version.get_minor()),
      _patch(version.get_patch()) {}

void OpenvinoVersion::read(std::istream& stream) {
    stream.read(reinterpret_cast<char*>(&_major), sizeof(_major));
    stream.read(reinterpret_cast<char*>(&_minor), sizeof(_minor));
    stream.read(reinterpret_cast<char*>(&_patch), sizeof(_patch));
}

void OpenvinoVersion::read(const ov::Tensor& tensor) {
    _major = *reinterpret_cast<const decltype(_major)*>(tensor.data<const char>());
    _minor = *reinterpret_cast<const decltype(_minor)*>(tensor.data<const char>() + sizeof(_major));
    _patch = *reinterpret_cast<const decltype(_patch)*>(tensor.data<const char>() + sizeof(_major) + sizeof(_minor));
}

void OpenvinoVersion::write(std::ostream& stream) {
    stream.write(reinterpret_cast<const char*>(&_major), sizeof(_major));
    stream.write(reinterpret_cast<const char*>(&_minor), sizeof(_minor));
    stream.write(reinterpret_cast<const char*>(&_patch), sizeof(_patch));
}

size_t OpenvinoVersion::get_openvino_version_size() const {
    return sizeof(_major) + sizeof(_minor) + sizeof(_patch);
}

MetadataBase::MetadataBase(uint32_t version, uint64_t blobDataSize) : _version(version), _blobDataSize(blobDataSize) {}

Metadata<METADATA_VERSION_2_0>::Metadata(uint64_t blobSize, std::optional<OpenvinoVersion> ovVersion)
    : MetadataBase{METADATA_VERSION_2_0, blobSize},
      _ovVersion{ovVersion.value_or(CURRENT_OPENVINO_VERSION)} {}

Metadata<METADATA_VERSION_2_1>::Metadata(uint64_t blobSize,
                                         std::optional<OpenvinoVersion> ovVersion,
                                         const std::optional<std::vector<uint64_t>> initSizes)
    : Metadata<METADATA_VERSION_2_0>{blobSize, ovVersion},
      _initSizes{initSizes} {
    _version = METADATA_VERSION_2_1;
}

Metadata<METADATA_VERSION_2_2>::Metadata(uint64_t blobSize,
                                         std::optional<OpenvinoVersion> ovVersion,
                                         const std::optional<std::vector<uint64_t>> initSizes,
                                         const std::optional<ov::Dimension> batchSize)
    : Metadata<METADATA_VERSION_2_1>{blobSize, ovVersion, initSizes},
      _batchSize{batchSize} {
    _version = METADATA_VERSION_2_2;
}

void Metadata<METADATA_VERSION_2_0>::read(std::istream& stream) {
    _ovVersion.read(stream);
}

void Metadata<METADATA_VERSION_2_0>::read(const ov::Tensor& tensor) {
    _ovVersion.read(tensor);
}

void Metadata<METADATA_VERSION_2_1>::read(std::istream& stream) {
    Metadata<METADATA_VERSION_2_0>::read(stream);

    uint64_t numberOfInits;
    stream.read(reinterpret_cast<char*>(&numberOfInits), sizeof(numberOfInits));

    if (numberOfInits) {
        _initSizes = std::vector<uint64_t>(numberOfInits);
        for (uint64_t initIndex = 0; initIndex < numberOfInits; ++initIndex) {
            stream.read(reinterpret_cast<char*>(&_initSizes->at(initIndex)), sizeof(_initSizes->at(initIndex)));
        }
    }
}

void Metadata<METADATA_VERSION_2_1>::read(const ov::Tensor& tensor) {
    Metadata<METADATA_VERSION_2_0>::read(tensor);
    auto roiTensor = ov::Tensor(tensor,
                                ov::Coordinate{sizeof(decltype(std::declval<OpenvinoVersion>().get_major())) +
                                               sizeof(decltype(std::declval<OpenvinoVersion>().get_minor())) +
                                               sizeof(decltype(std::declval<OpenvinoVersion>().get_patch()))},
                                ov::Coordinate{tensor.get_byte_size()});

    uint64_t numberOfInits;
    numberOfInits = *reinterpret_cast<const decltype(numberOfInits)*>(roiTensor.data<const char>());

    if (numberOfInits) {
        _initSizes = std::vector<uint64_t>(numberOfInits);
        for (uint64_t initIndex = 0; initIndex < numberOfInits; ++initIndex) {
            _initSizes->at(initIndex) =
                reinterpret_cast<const std::remove_reference_t<decltype(_initSizes->at(initIndex))>*>(
                    roiTensor.data<const char>() + sizeof(numberOfInits))[initIndex];
        }
    }
}

void Metadata<METADATA_VERSION_2_2>::read(std::istream& stream) {
    Metadata<METADATA_VERSION_2_1>::read(stream);

    stream.read(reinterpret_cast<char*>(&_batchSize), sizeof(_batchSize));
}

void Metadata<METADATA_VERSION_2_2>::read(const ov::Tensor& tensor) {
    Metadata<METADATA_VERSION_2_1>::read(tensor);
    auto roiTensor = ov::Tensor(tensor,
                                ov::Coordinate{sizeof(decltype(std::declval<OpenvinoVersion>().get_major())) +
                                               sizeof(decltype(std::declval<OpenvinoVersion>().get_minor())) +
                                               sizeof(decltype(std::declval<OpenvinoVersion>().get_patch()))},
                                ov::Coordinate{tensor.get_byte_size()});

    _batchSize = *reinterpret_cast<const decltype(_batchSize)*>(roiTensor.data<const char>());
}

void MetadataBase::append_padding_blob_size_and_magic(std::ostream& stream) {
    size_t metadataSize = get_metadata_size() + sizeof(_blobDataSize) + MAGIC_BYTES.size();
    size_t size = utils::align_size_to_standard_page_size(metadataSize);
    size_t paddingSize = size - metadataSize;
    if (paddingSize > 0) {
        std::fill_n(std::ostream_iterator<char>(stream), paddingSize, 0);
    }

    stream.write(reinterpret_cast<const char*>(&_blobDataSize), sizeof(_blobDataSize));
    stream.write(MAGIC_BYTES.data(), MAGIC_BYTES.size());
}

void Metadata<METADATA_VERSION_2_0>::write(std::ostream& stream) {
    stream.write(reinterpret_cast<const char*>(&_version), sizeof(_version));
    _ovVersion.write(stream);
}

void Metadata<METADATA_VERSION_2_1>::write(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_0>::write(stream);

    _numberOfInits = _initSizes.has_value() ? _initSizes->size() : 0;
    stream.write(reinterpret_cast<const char*>(&_numberOfInits), sizeof(_numberOfInits));

    if (_initSizes.has_value()) {
        for (uint64_t initSize : _initSizes.value()) {
            stream.write(reinterpret_cast<const char*>(&initSize), sizeof(initSize));
        }
    }

    append_padding_blob_size_and_magic(stream);
}

void Metadata<METADATA_VERSION_2_2>::write(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_1>::write(stream);

    stream.write(reinterpret_cast<const char*>(&_batchSize), sizeof(_batchSize));

    append_padding_blob_size_and_magic(stream);
}

std::unique_ptr<MetadataBase> create_metadata(uint32_t version, uint64_t blobSize) {
    if (MetadataBase::get_major(version) == CURRENT_METADATA_MAJOR_VERSION &&
        MetadataBase::get_minor(version) > CURRENT_METADATA_MINOR_VERSION) {
        return std::make_unique<Metadata<CURRENT_METADATA_VERSION>>(blobSize, std::nullopt);
    }

    switch (version) {
    case METADATA_VERSION_2_0:
        return std::make_unique<Metadata<METADATA_VERSION_2_0>>(blobSize, std::nullopt);
    case METADATA_VERSION_2_1:
        return std::make_unique<Metadata<METADATA_VERSION_2_1>>(blobSize, std::nullopt);
    case METADATA_VERSION_2_2:
        return std::make_unique<Metadata<METADATA_VERSION_2_1>>(blobSize, std::nullopt);
    default:
        OPENVINO_THROW("Metadata version is not supported!");
    }
}

bool Metadata<METADATA_VERSION_2_0>::is_compatible() {
    auto logger = Logger::global().clone("NPUBlobMetadata");
    // checking if we can import the blob
    if (_ovVersion != CURRENT_OPENVINO_VERSION) {
        logger.error("Imported blob OpenVINO version: %d.%d.%d, but the current OpenVINO version is: %d.%d.%d",
                     _ovVersion.get_major(),
                     _ovVersion.get_minor(),
                     _ovVersion.get_patch(),
                     OPENVINO_VERSION_MAJOR,
                     OPENVINO_VERSION_MINOR,
                     OPENVINO_VERSION_PATCH);
        return false;
    }
    return true;
}

std::streampos MetadataBase::getFileSize(std::istream& stream) {
    auto log = intel_npu::Logger::global().clone("getFileSize");
    if (!stream) {
        OPENVINO_THROW("Stream is in bad status! Please check the passed stream status!");
    }

    if (dynamic_cast<ov::SharedStreamBuffer*>(stream.rdbuf()) != nullptr) {
        return stream.rdbuf()->in_avail();
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

std::unique_ptr<MetadataBase> read_metadata_from(std::istream& stream) {
    size_t magicBytesSize = MAGIC_BYTES.size();
    std::string blobMagicBytes;
    blobMagicBytes.resize(magicBytesSize);

    std::streampos currentStreamPos = stream.tellg(), streamSize = MetadataBase::getFileSize(stream);
    stream.seekg(streamSize - std::streampos(magicBytesSize), std::ios::cur);
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
    } catch (const std::exception& ex) {
        OPENVINO_THROW(ex.what(),
                       "Imported blob metadata version: ",
                       MetadataBase::get_major(metaVersion),
                       ".",
                       MetadataBase::get_minor(metaVersion),
                       " but the current version is: ",
                       CURRENT_METADATA_MAJOR_VERSION,
                       ".",
                       CURRENT_METADATA_MINOR_VERSION);
    } catch (...) {
        OPENVINO_THROW("Unexpected exception while reading blob NPU metadata");
    }
    stream.seekg(-stream.tellg() + currentStreamPos, std::ios::cur);

    return storedMeta;
}

std::unique_ptr<MetadataBase> read_metadata_from(const ov::Tensor& tensor) {
    size_t magicBytesSize = MAGIC_BYTES.size();
    std::string_view blobMagicBytes(tensor.data<const char>() + tensor.get_byte_size() - magicBytesSize,
                                    magicBytesSize);

    if (MAGIC_BYTES != blobMagicBytes) {
        OPENVINO_THROW("Blob is missing NPU metadata!");
    }

    uint64_t blobDataSize;
    blobDataSize = *reinterpret_cast<const decltype(blobDataSize)*>(tensor.data<const char>() + tensor.get_byte_size() -
                                                                    magicBytesSize - sizeof(blobDataSize));

    uint32_t metaVersion;
    metaVersion = *reinterpret_cast<const decltype(metaVersion)*>(tensor.data<const char>() + blobDataSize);

    std::unique_ptr<MetadataBase> storedMeta;
    try {
        auto roiTensor = ov::Tensor(tensor,
                                    ov::Coordinate{blobDataSize + sizeof(metaVersion)},
                                    ov::Coordinate{tensor.get_byte_size()});
        storedMeta = create_metadata(metaVersion, blobDataSize);
        storedMeta->read(roiTensor);
    } catch (const std::exception& ex) {
        OPENVINO_THROW(ex.what(),
                       "Imported blob metadata version: ",
                       MetadataBase::get_major(metaVersion),
                       ".",
                       MetadataBase::get_minor(metaVersion),
                       " but the current version is: ",
                       CURRENT_METADATA_MAJOR_VERSION,
                       ".",
                       CURRENT_METADATA_MINOR_VERSION);
    } catch (...) {
        OPENVINO_THROW("Unexpected exception while reading blob NPU metadata");
    }

    return storedMeta;
}

uint64_t MetadataBase::get_blob_size() const {
    return _blobDataSize;
}

std::optional<std::vector<uint64_t>> Metadata<METADATA_VERSION_2_0>::get_init_sizes() const {
    return std::nullopt;
}

std::optional<std::vector<uint64_t>> Metadata<METADATA_VERSION_2_1>::get_init_sizes() const {
    return _initSizes;
}

std::optional<ov::Dimension> Metadata<METADATA_VERSION_2_2>::get_batch_size() const {
    return _batchSize;
}

size_t Metadata<METADATA_VERSION_2_0>::get_metadata_size() const {
    return sizeof(_version) + _ovVersion.get_openvino_version_size();
}

size_t Metadata<METADATA_VERSION_2_1>::get_metadata_size() const {
    size_t metadataSize = Metadata<METADATA_VERSION_2_0>::get_metadata_size() + sizeof(_numberOfInits);

    if (_initSizes.has_value()) {
        for (size_t initSize : _initSizes.value()) {
            metadataSize += sizeof(initSize);
        }
    }

    return metadataSize;
}

}  // namespace intel_npu
