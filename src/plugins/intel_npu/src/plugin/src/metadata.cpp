// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "metadata.hpp"

#include <cstring>
#include <iterator>
#include <optional>

#include "intel_npu/utils/utils.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/variant_visitor.hpp"

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

MetadataBase::MetadataBase(uint32_t version, uint64_t blobDataSize)
    : _version(version),
      _blobDataSize(blobDataSize),
      _logger("NPUBlobMetadata", Logger::global().level()),
      _source() {}

Metadata<METADATA_VERSION_2_0>::Metadata(uint64_t blobSize, const std::optional<OpenvinoVersion>& ovVersion)
    : MetadataBase{METADATA_VERSION_2_0, blobSize},
      _ovVersion{ovVersion.value_or(CURRENT_OPENVINO_VERSION)} {}

Metadata<METADATA_VERSION_2_1>::Metadata(uint64_t blobSize,
                                         const std::optional<OpenvinoVersion>& ovVersion,
                                         const std::optional<std::vector<uint64_t>>& initSizes)
    : Metadata<METADATA_VERSION_2_0>{blobSize, ovVersion},
      _initSizes{initSizes} {
    _version = METADATA_VERSION_2_1;
}

Metadata<METADATA_VERSION_2_2>::Metadata(uint64_t blobSize,
                                         std::optional<OpenvinoVersion> ovVersion,
                                         const std::optional<std::vector<uint64_t>> initSizes,
                                         const std::optional<int64_t> batchSize)
    : Metadata<METADATA_VERSION_2_1>{blobSize, ovVersion, initSizes},
      _batchSize{batchSize} {
    _version = METADATA_VERSION_2_2;
}

Metadata<METADATA_VERSION_2_3>::Metadata(uint64_t blobSize,
                                         const std::optional<OpenvinoVersion>& ovVersion,
                                         const std::optional<std::vector<uint64_t>>& initSizes,
                                         const std::optional<int64_t> batchSize,
                                         const std::optional<std::vector<ov::Layout>>& inputLayouts,
                                         const std::optional<std::vector<ov::Layout>>& outputLayouts)
    : Metadata<METADATA_VERSION_2_2>{blobSize, ovVersion, initSizes, batchSize},
      _inputLayouts{inputLayouts},
      _outputLayouts{outputLayouts} {
    _version = METADATA_VERSION_2_3;
}

void MetadataBase::read(std::istream& tensor) {
    _source = Source(tensor);
    read();
}

void MetadataBase::read(const ov::Tensor& tensor) {
    _source = Source(tensor);
    read();
}

void MetadataBase::read_data_from_source(char* destination, const size_t size) {
    if (const std::reference_wrapper<std::istream>* stream =
            std::get_if<std::reference_wrapper<std::istream>>(&_source)) {
        stream->get().read(destination, size);
    } else if (const std::reference_wrapper<const ov::Tensor>* tensor =
                   std::get_if<std::reference_wrapper<const ov::Tensor>>(&_source)) {
        std::memcpy(destination, tensor->get().data<const char>() + _cursorOffset, size);
        _cursorOffset += size;
    } else {
        OPENVINO_THROW("No blob has been provided to NPU plugin's metadata reader.");
    }
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

void Metadata<METADATA_VERSION_2_0>::read() {
    if (const std::reference_wrapper<std::istream>* source =
            std::get_if<std::reference_wrapper<std::istream>>(&_source)) {
        _ovVersion.read(*source);
    } else if (const std::reference_wrapper<const ov::Tensor>* source =
                   std::get_if<std::reference_wrapper<const ov::Tensor>>(&_source)) {
        _ovVersion.read(*source);
        _cursorOffset = _ovVersion.get_openvino_version_size();
    } else {
        OPENVINO_THROW("No blob has been provided to NPU plugin's metadata reader.");
    }
}

void Metadata<METADATA_VERSION_2_1>::read() {
    Metadata<METADATA_VERSION_2_0>::read();

    uint64_t numberOfInits;
    read_data_from_source(reinterpret_cast<char*>(&numberOfInits), sizeof(numberOfInits));

    if (numberOfInits) {
        _initSizes = std::vector<uint64_t>(numberOfInits);
        for (uint64_t initIndex = 0; initIndex < numberOfInits; ++initIndex) {
            read_data_from_source(reinterpret_cast<char*>(&_initSizes->at(initIndex)),
                                  sizeof(_initSizes->at(initIndex)));
        }
    }
}

void Metadata<METADATA_VERSION_2_2>::read() {
    Metadata<METADATA_VERSION_2_1>::read();

    int64_t batchSize;
    read_data_from_source(reinterpret_cast<char*>(&batchSize), sizeof(batchSize));

    _batchSize = batchSize != 0 ? std::optional(batchSize) : std::nullopt;
}

void Metadata<METADATA_VERSION_2_3>::read() {
    Metadata<METADATA_VERSION_2_2>::read();

    uint64_t numberOfInputLayouts, numberOfOutputLayouts;
    read_data_from_source(reinterpret_cast<char*>(&numberOfInputLayouts), sizeof(numberOfInputLayouts));
    read_data_from_source(reinterpret_cast<char*>(&numberOfOutputLayouts), sizeof(numberOfOutputLayouts));

    const auto readNLayouts = [&](const uint64_t numberOfLayouts, const char* loggerAddition) {
        std::optional<std::vector<ov::Layout>> layouts = std::nullopt;
        if (!numberOfLayouts) {
            return layouts;
        }

        uint16_t stringLength;
        layouts = std::vector<ov::Layout>();
        layouts->reserve(numberOfLayouts);
        for (uint64_t layoutIndex = 0; layoutIndex < numberOfLayouts; ++layoutIndex) {
            read_data_from_source(reinterpret_cast<char*>(&stringLength), sizeof(stringLength));

            std::string layoutString(stringLength, 0);
            read_data_from_source(const_cast<char*>(layoutString.c_str()), stringLength);

            try {
                layouts->push_back(ov::Layout(std::move(layoutString)));
            } catch (const ov::Exception&) {
                _logger.warning("Error encountered while constructing an ov::Layout object. %s index: %d. Value "
                                "read from blob: %s. A default value will be used instead.",
                                loggerAddition,
                                layoutIndex,
                                layoutString.c_str());
                layouts->push_back(ov::Layout());
            }
        }
        return layouts;
    };

    _inputLayouts = readNLayouts(numberOfInputLayouts, "Input");
    _outputLayouts = readNLayouts(numberOfOutputLayouts, "Output");
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
}

void Metadata<METADATA_VERSION_2_2>::write(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_1>::write(stream);

    int64_t batchValue = _batchSize.value_or(0);
    stream.write(reinterpret_cast<const char*>(&batchValue), sizeof(batchValue));
}

void Metadata<METADATA_VERSION_2_3>::write(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_2>::write(stream);

    const uint64_t numberOfInputLayouts = _inputLayouts.has_value() ? _inputLayouts->size() : 0;
    const uint64_t numberOfOutputLayouts = _outputLayouts.has_value() ? _outputLayouts->size() : 0;
    stream.write(reinterpret_cast<const char*>(&numberOfInputLayouts), sizeof(numberOfInputLayouts));
    stream.write(reinterpret_cast<const char*>(&numberOfOutputLayouts), sizeof(numberOfOutputLayouts));

    const auto writeLayouts = [&](const std::optional<std::vector<ov::Layout>>& layouts) {
        if (layouts.has_value()) {
            for (const ov::Layout& layout : layouts.value()) {
                const std::string layoutString = layout.to_string();
                const uint16_t stringLength = static_cast<uint16_t>(layoutString.size());
                stream.write(reinterpret_cast<const char*>(&stringLength), sizeof(stringLength));
                stream.write(layoutString.c_str(), stringLength);
            }
        }
    };

    writeLayouts(_inputLayouts);
    writeLayouts(_outputLayouts);

    append_padding_blob_size_and_magic(stream);
}

std::unique_ptr<MetadataBase> create_metadata(uint32_t version, uint64_t blobSize) {
    uint16_t major = MetadataBase::get_major(version), minor = MetadataBase::get_minor(version);
    if (major != CURRENT_METADATA_MAJOR_VERSION || minor > CURRENT_METADATA_MINOR_VERSION) {
        OPENVINO_THROW("Metadata version is not supported! Imported blob metadata version: ",
                       major,
                       ".",
                       minor,
                       " but the current version is: ",
                       CURRENT_METADATA_MAJOR_VERSION,
                       ".",
                       CURRENT_METADATA_MINOR_VERSION);
    }

    switch (version) {
    case METADATA_VERSION_2_0:
        return std::make_unique<Metadata<METADATA_VERSION_2_0>>(blobSize);
    case METADATA_VERSION_2_1:
        return std::make_unique<Metadata<METADATA_VERSION_2_1>>(blobSize);
    case METADATA_VERSION_2_2:
        return std::make_unique<Metadata<METADATA_VERSION_2_2>>(blobSize);
    case METADATA_VERSION_2_3:
        return std::make_unique<Metadata<METADATA_VERSION_2_3>>(blobSize);
    default:
        return nullptr;
    }
}

std::streampos MetadataBase::getFileSize(std::istream& stream) {
    auto log = Logger::global().clone("getFileSize");
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
        OPENVINO_THROW("Can't read NPU metadata: ", ex.what());
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
        const ov::Tensor roiTensor(tensor,
                                   ov::Coordinate{blobDataSize + sizeof(metaVersion)},
                                   ov::Coordinate{tensor.get_byte_size()});
        storedMeta = create_metadata(metaVersion, blobDataSize);
        storedMeta->read(roiTensor);
    } catch (const std::exception& ex) {
        OPENVINO_THROW("Can't read NPU metadata: ", ex.what());
    } catch (...) {
        OPENVINO_THROW("Unexpected exception while reading blob NPU metadata");
    }

    return storedMeta;
}

uint64_t MetadataBase::get_blob_size() const {
    return _blobDataSize;
}

std::optional<std::vector<uint64_t>> MetadataBase::get_init_sizes() const {
    return std::nullopt;
}

std::optional<int64_t> MetadataBase::get_batch_size() const {
    return std::nullopt;
}

std::optional<std::vector<ov::Layout>> MetadataBase::get_input_layouts() const {
    return std::nullopt;
}

std::optional<std::vector<ov::Layout>> MetadataBase::get_output_layouts() const {
    return std::nullopt;
}

std::optional<std::vector<uint64_t>> Metadata<METADATA_VERSION_2_1>::get_init_sizes() const {
    return _initSizes;
}

std::optional<int64_t> Metadata<METADATA_VERSION_2_2>::get_batch_size() const {
    return _batchSize;
}

std::optional<std::vector<ov::Layout>> Metadata<METADATA_VERSION_2_3>::get_input_layouts() const {
    return _inputLayouts;
}

std::optional<std::vector<ov::Layout>> Metadata<METADATA_VERSION_2_3>::get_output_layouts() const {
    return _outputLayouts;
}

size_t Metadata<METADATA_VERSION_2_0>::get_metadata_size() const {
    return sizeof(_version) + _ovVersion.get_openvino_version_size();
}

size_t Metadata<METADATA_VERSION_2_1>::get_metadata_size() const {
    size_t metadataSize = Metadata<METADATA_VERSION_2_0>::get_metadata_size() + sizeof(_numberOfInits);

    if (_initSizes.has_value()) {
        metadataSize += _initSizes->size() * sizeof(uint64_t);
    }

    return metadataSize;
}

size_t Metadata<METADATA_VERSION_2_2>::get_metadata_size() const {
    size_t metadataSize = Metadata<METADATA_VERSION_2_1>::get_metadata_size() + sizeof(int64_t);

    return metadataSize;
}

size_t Metadata<METADATA_VERSION_2_3>::get_metadata_size() const {
    size_t metadataSize = Metadata<METADATA_VERSION_2_2>::get_metadata_size();
    // Number of input layouts & number of output layouts
    metadataSize += 2 * sizeof(uint64_t);

    if (_inputLayouts.has_value()) {
        for (const ov::Layout& layout : _inputLayouts.value()) {
            // Length followed by the layout value as string
            metadataSize += sizeof(uint16_t) + layout.to_string().size();
        }
    }
    if (_outputLayouts.has_value()) {
        for (const ov::Layout& layout : _outputLayouts.value()) {
            metadataSize += sizeof(uint16_t) + layout.to_string().size();
        }
    }

    return metadataSize;
}

}  // namespace intel_npu
