// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "metadata.hpp"

#include <cstring>
#include <optional>
#include <string>

#include "intel_npu/compat_string_parser.hpp"
#include "openvino/runtime/shared_buffer.hpp"

namespace {

template <typename T>
void write_text_field(std::ostream& stream, std::string_view key, const T& value) {
    if (stream.tellp() != std::streampos(0)) {
        stream << ';';
    }
    stream << key << '=' << value;
}

}  // namespace

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
                                         const std::optional<std::vector<uint64_t>>& initSizes,
                                         const std::optional<int64_t>& batchSize)
    : Metadata<METADATA_VERSION_2_1>{blobSize, ovVersion, initSizes},
      _batchSize{batchSize} {
    _version = METADATA_VERSION_2_2;
}

Metadata<METADATA_VERSION_2_3>::Metadata(uint64_t blobSize,
                                         const std::optional<OpenvinoVersion>& ovVersion,
                                         const std::optional<std::vector<uint64_t>>& initSizes,
                                         const std::optional<int64_t>& batchSize,
                                         const std::optional<std::vector<ov::Layout>>& inputLayouts,
                                         const std::optional<std::vector<ov::Layout>>& outputLayouts)
    : Metadata<METADATA_VERSION_2_2>{blobSize, ovVersion, initSizes, batchSize},
      _inputLayouts{inputLayouts},
      _outputLayouts{outputLayouts} {
    _version = METADATA_VERSION_2_3;
}

Metadata<METADATA_VERSION_2_4>::Metadata(uint64_t blobSize,
                                         const std::optional<OpenvinoVersion>& ovVersion,
                                         const std::optional<std::vector<uint64_t>>& initSizes,
                                         const std::optional<int64_t>& batchSize,
                                         const std::optional<std::vector<ov::Layout>>& inputLayouts,
                                         const std::optional<std::vector<ov::Layout>>& outputLayouts,
                                         const std::optional<uint32_t>& compilerVersion)
    : Metadata<METADATA_VERSION_2_3>{blobSize, ovVersion, initSizes, batchSize, inputLayouts, outputLayouts},
      _compilerVersion{compilerVersion} {
    _version = METADATA_VERSION_2_4;
}

Metadata<METADATA_VERSION_2_5>::Metadata(uint64_t blobSize,
                                         const std::optional<OpenvinoVersion>& ovVersion,
                                         const std::optional<std::vector<uint64_t>>& initSizes,
                                         const std::optional<int64_t>& batchSize,
                                         const std::optional<std::vector<ov::Layout>>& inputLayouts,
                                         const std::optional<std::vector<ov::Layout>>& outputLayouts,
                                         const std::optional<uint32_t>& compilerVersion,
                                         const std::optional<uint64_t>& blobSizeAfterEncryption)
    : Metadata<METADATA_VERSION_2_4>{blobSizeAfterEncryption.has_value() ? blobSizeAfterEncryption.value() : blobSize,
                                     ovVersion,
                                     initSizes,
                                     batchSize,
                                     inputLayouts,
                                     outputLayouts,
                                     compilerVersion},
      _isEncryptedBlob{blobSizeAfterEncryption.has_value()} {
    _version = METADATA_VERSION_2_5;
}

void MetadataBase::read(std::istream& tensor) {
    _source = Source(tensor);
    read();
}

void MetadataBase::read(const ov::Tensor& tensor) {
    _source = Source(tensor);
    read();
}

void MetadataBase::read_as_text(const ov::Tensor& tensor) {
    const std::string_view input(tensor.data<const char>(), tensor.get_byte_size());
    _parser.parse(input, metadataTextAttributes);
    read_as_text();
}

void MetadataBase::read_data_from_source(char* destination, const size_t size) {
    if (const std::reference_wrapper<std::istream>* stream =
            std::get_if<std::reference_wrapper<std::istream>>(&_source)) {
        stream->get().read(destination, size);
    } else if (const std::reference_wrapper<const ov::Tensor>* tensor =
                   std::get_if<std::reference_wrapper<const ov::Tensor>>(&_source)) {
        const size_t available = tensor->get().get_byte_size();
        const size_t remaining = (_cursorOffset <= available) ? available - _cursorOffset : 0;
        if (size > remaining) {
            OPENVINO_THROW("NPU metadata: attempted to read ",
                           size,
                           " bytes at offset ",
                           _cursorOffset,
                           " but only ",
                           remaining,
                           " bytes remain in the metadata buffer.");
        }
        std::memcpy(destination, tensor->get().data<const char>() + _cursorOffset, size);
        _cursorOffset += size;
    } else {
        OPENVINO_THROW("No blob has been provided to NPU plugin's metadata reader.");
    }
}

void MetadataBase::append_blob_size_and_magic(std::ostream& stream) {
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

void Metadata<METADATA_VERSION_2_4>::read() {
    Metadata<METADATA_VERSION_2_3>::read();

    uint32_t compilerVersion;
    read_data_from_source(reinterpret_cast<char*>(&compilerVersion), sizeof(compilerVersion));
    _compilerVersion = compilerVersion != 0 ? std::optional(compilerVersion) : std::nullopt;
}

void Metadata<METADATA_VERSION_2_5>::read() {
    Metadata<METADATA_VERSION_2_4>::read();

    uint8_t isEncryptedBlob;
    read_data_from_source(reinterpret_cast<char*>(&isEncryptedBlob), sizeof(isEncryptedBlob));

    _isEncryptedBlob = isEncryptedBlob;
}

void Metadata<METADATA_VERSION_2_0>::read_as_text() {
    const auto& attrs = _parser.getAttributes();
    const auto it = attrs.find(MetadataTextKeys::OV);
    if (it == attrs.end()) {
        OPENVINO_THROW("Human-readable metadata missing '" + std::string(MetadataTextKeys::OV) + "' field.");
    }
    const std::string& s = it->second;
    const size_t dot1 = s.find('.');
    if (dot1 == std::string::npos) {
        OPENVINO_THROW("Human-readable metadata: '" + std::string(MetadataTextKeys::OV) +
                       "' is not in MAJOR.MINOR.PATCH format: " + s);
    }
    const size_t dot2 = s.find('.', dot1 + 1);
    if (dot2 == std::string::npos || dot2 == dot1 + 1 || dot2 + 1 >= s.size()) {
        OPENVINO_THROW("Human-readable metadata: '" + std::string(MetadataTextKeys::OV) +
                       "' is not in MAJOR.MINOR.PATCH format: " + s);
    }
    _ovVersion = OpenvinoVersion(static_cast<uint16_t>(std::stoul(s.substr(0, dot1))),
                                 static_cast<uint16_t>(std::stoul(s.substr(dot1 + 1, dot2 - dot1 - 1))),
                                 static_cast<uint16_t>(std::stoul(s.substr(dot2 + 1))));
}

void Metadata<METADATA_VERSION_2_1>::read_as_text() {
    Metadata<METADATA_VERSION_2_0>::read_as_text();

    const auto& attrs = _parser.getAttributes();
    const auto it = attrs.find(MetadataTextKeys::WS_INITS);
    if (it == attrs.end()) {
        return;
    }
    if (it->second != "TRUE") {
        OPENVINO_THROW("Human-readable metadata: '" + std::string(MetadataTextKeys::WS_INITS) +
                       "' must be 'TRUE' when present; got: " + it->second);
    }
    _initSizes = std::vector<uint64_t>{};
}

void Metadata<METADATA_VERSION_2_2>::read_as_text() {
    Metadata<METADATA_VERSION_2_1>::read_as_text();

    const auto& attrs = _parser.getAttributes();
    const auto it = attrs.find(MetadataTextKeys::BATCH);
    if (it == attrs.end()) {
        return;
    }
    const int64_t batchValue = std::stoll(it->second);
    _batchSize = (batchValue != 0) ? std::optional<int64_t>(batchValue) : std::nullopt;
}

void Metadata<METADATA_VERSION_2_3>::read_as_text() {
    Metadata<METADATA_VERSION_2_2>::read_as_text();
}

void Metadata<METADATA_VERSION_2_4>::read_as_text() {
    Metadata<METADATA_VERSION_2_3>::read_as_text();
    _compilerVersion = std::nullopt;
}

void Metadata<METADATA_VERSION_2_5>::read_as_text() {
    Metadata<METADATA_VERSION_2_4>::read_as_text();
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
}

void Metadata<METADATA_VERSION_2_4>::write(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_3>::write(stream);

    uint32_t compilerVersion = _compilerVersion.value_or(0);
    stream.write(reinterpret_cast<const char*>(&compilerVersion), sizeof(compilerVersion));
}

void Metadata<METADATA_VERSION_2_5>::write(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_4>::write(stream);

    const uint8_t isEncryptedBlob = _isEncryptedBlob.value_or(false);
    stream.write(reinterpret_cast<const char*>(&isEncryptedBlob), sizeof(isEncryptedBlob));

    append_blob_size_and_magic(stream);
}

void Metadata<METADATA_VERSION_2_0>::write_as_text(std::ostream& stream) {
    const uint16_t meta_major = MetadataBase::get_major(_version);
    const uint16_t meta_minor = MetadataBase::get_minor(_version);
    write_text_field(stream, MetadataTextKeys::META, std::to_string(meta_major) + "." + std::to_string(meta_minor));
    write_text_field(stream,
                     MetadataTextKeys::OV,
                     std::to_string(OPENVINO_VERSION_MAJOR) + "." + std::to_string(OPENVINO_VERSION_MINOR) + "." +
                         std::to_string(OPENVINO_VERSION_PATCH));
}

void Metadata<METADATA_VERSION_2_1>::write_as_text(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_0>::write_as_text(stream);

    if (_initSizes.has_value() && !_initSizes->empty()) {
        write_text_field(stream, MetadataTextKeys::WS_INITS, "TRUE");
    }
}

void Metadata<METADATA_VERSION_2_2>::write_as_text(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_1>::write_as_text(stream);

    if (_batchSize.has_value() && _batchSize.value() > 0) {
        write_text_field(stream, MetadataTextKeys::BATCH, _batchSize.value());
    }
}

void Metadata<METADATA_VERSION_2_3>::write_as_text(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_2>::write_as_text(stream);
}

void Metadata<METADATA_VERSION_2_4>::write_as_text(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_3>::write_as_text(stream);
}

void Metadata<METADATA_VERSION_2_5>::write_as_text(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_4>::write_as_text(stream);
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
    case METADATA_VERSION_2_4:
        return std::make_unique<Metadata<METADATA_VERSION_2_4>>(blobSize);
    case METADATA_VERSION_2_5:
        return std::make_unique<Metadata<METADATA_VERSION_2_5>>(blobSize);
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

std::unique_ptr<MetadataBase> read_as_text(const ov::Tensor& tensor) {
    const std::string_view input(tensor.data<const char>(), tensor.get_byte_size());
    compat::Parser parser;
    try {
        parser.parse(input, metadataTextAttributes);
    } catch (const std::exception& ex) {
        OPENVINO_THROW("NPU compatibility string is malformed: ", ex.what());
    }

    const auto& attrs = parser.getAttributes();
    const auto it = attrs.find(MetadataTextKeys::META);
    if (it == attrs.end()) {
        OPENVINO_THROW("NPU compatibility string is malformed: missing metadata version");
    }

    const std::string& versionStr = it->second;
    const size_t dot = versionStr.find('.');
    if (dot == std::string::npos) {
        OPENVINO_THROW("Invalid human-readable metadata: malformed version field");
    }

    const uint16_t major = static_cast<uint16_t>(std::stoul(versionStr.substr(0, dot)));
    const uint16_t minor = static_cast<uint16_t>(std::stoul(versionStr.substr(dot + 1)));
    const uint32_t metaVersion = MetadataBase::make_version(major, minor);

    std::unique_ptr<MetadataBase> storedMeta;
    try {
        storedMeta = create_metadata(metaVersion, 0);
        storedMeta->read_as_text(tensor);
    } catch (const std::exception& ex) {
        OPENVINO_THROW("Can't read NPU human-readable metadata: ", ex.what());
    } catch (...) {
        OPENVINO_THROW("Unexpected exception while reading NPU human-readable metadata");
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

std::optional<uint32_t> MetadataBase::get_compiler_version() const {
    return std::nullopt;
}

std::optional<bool> MetadataBase::is_encrypted_blob() const {
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

std::optional<uint32_t> Metadata<METADATA_VERSION_2_4>::get_compiler_version() const {
    return _compilerVersion;
}

std::optional<bool> Metadata<METADATA_VERSION_2_5>::is_encrypted_blob() const {
    return _isEncryptedBlob;
}

}  // namespace intel_npu
