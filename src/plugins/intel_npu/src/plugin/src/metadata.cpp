// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "metadata.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <optional>
#include <string>

#include "intel_npu/compat_string_parser.hpp"
#include "openvino/runtime/shared_buffer.hpp"

namespace {

// Compiler payload size + magic bytes
constexpr size_t FOOTER_SIZE = sizeof(uint64_t) + intel_npu::MAGIC_BYTES.size();
// Metadata version + compiler payload size + magic bytes
constexpr size_t MINIMUM_BLOB_SIZE = sizeof(uint32_t) + FOOTER_SIZE;
constexpr size_t SIZE_OF_INIT_SCHEDULE_SIZE = sizeof(uint64_t);
constexpr size_t SIZE_OF_LAYOUT_SIZE = sizeof(uint16_t);

constexpr std::string_view MISSING_METADATA_MESSAGE = "The blob is missing the NPU metadata!";
constexpr std::string_view BLOB_TOO_SMALL_MESSAGE =
    "The blob received for parsing is too small to contain all mandatory information. Blob size: ";
constexpr std::string_view INVALID_PAYLOAD_SIZE_MESSAGE =
    "The size of the compiler payload parsed from the blob is greater "
    "than the size of the blob. Compiler payload size: ";
constexpr std::string_view MISSING_BLOB_MESSAGE = "No blob has been provided to NPU plugin's metadata reader.";
constexpr std::string_view STREAM_BAD_STATUS_MESSAGE = "The stream is in bad status";

template <typename T>
void write_text_field(std::ostream& stream, std::string_view key, const T& value) {
    if (stream.tellp() != std::streampos(0)) {
        stream << ';';
    }
    stream << key << '=' << value;
}

std::vector<uint16_t> parse_version(std::string_view sv) {
    const auto hasOnlyDigits = [](std::string_view sv) {
        return !sv.empty() && std::all_of(sv.begin(), sv.end(), [](unsigned char c) {
            return std::isdigit(c);
        });
    };

    std::vector<uint16_t> parts;
    std::string_view remaining = sv;
    while (true) {
        const size_t dot = remaining.find('.');
        const std::string_view part = remaining.substr(0, dot);
        if (!hasOnlyDigits(part)) {
            OPENVINO_THROW("Invalid version '",
                           sv,
                           "': version must meet the format MAJOR.MINOR.PATCH with numeric components");
        }
        parts.push_back(static_cast<uint16_t>(std::stoul(std::string(part))));
        if (dot == std::string_view::npos) {
            break;
        }
        remaining = remaining.substr(dot + 1);
        if (remaining.empty()) {
            OPENVINO_THROW("Invalid version '", sv, "': trailing dot");
        }
    }
    return parts;
}

/**
 * @return The size of the underlying buffer, from the beginning of the stream to its end.
 */
size_t get_stream_total_size(std::istream& stream) {
    OPENVINO_ASSERT(stream, STREAM_BAD_STATUS_MESSAGE);

    const std::streampos backupCursor = stream.tellg();
    stream.seekg(0, std::ios_base::end);
    const std::streampos streamEnd = stream.tellg();
    stream.seekg(backupCursor, std::ios_base::beg);

    return streamEnd;
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

std::optional<BlobType> MetadataBase::get_blob_type() const {
    return std::nullopt;
}

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

Metadata<METADATA_VERSION_2_6>::Metadata(uint64_t blobSize,
                                         const std::optional<OpenvinoVersion>& ovVersion,
                                         const std::optional<std::vector<uint64_t>>& initSizes,
                                         const std::optional<int64_t> batchSize,
                                         const std::optional<std::vector<ov::Layout>>& inputLayouts,
                                         const std::optional<std::vector<ov::Layout>>& outputLayouts,
                                         const std::optional<uint32_t> compilerVersion,
                                         const std::optional<uint64_t>& blobSizeAfterEncryption,
                                         const std::optional<std::string_view> compatibilityDescriptor)
    : Metadata<METADATA_VERSION_2_5>{blobSize,
                                     ovVersion,
                                     initSizes,
                                     batchSize,
                                     inputLayouts,
                                     outputLayouts,
                                     compilerVersion,
                                     blobSizeAfterEncryption},
      _compatibilityDescriptor{compatibilityDescriptor} {
    _version = METADATA_VERSION_2_6;
}

Metadata<METADATA_VERSION_2_7>::Metadata(uint64_t blobSize,
                                         const std::optional<OpenvinoVersion>& ovVersion,
                                         const std::optional<std::vector<uint64_t>>& initSizes,
                                         const std::optional<int64_t> batchSize,
                                         const std::optional<std::vector<ov::Layout>>& inputLayouts,
                                         const std::optional<std::vector<ov::Layout>>& outputLayouts,
                                         const std::optional<uint32_t> compilerVersion,
                                         const std::optional<uint64_t>& blobSizeAfterEncryption,
                                         const std::optional<std::string_view> compatibilityDescriptor,
                                         BlobType blobType)
    : Metadata<METADATA_VERSION_2_6>{blobSize,
                                     ovVersion,
                                     initSizes,
                                     batchSize,
                                     inputLayouts,
                                     outputLayouts,
                                     compilerVersion,
                                     blobSizeAfterEncryption,
                                     compatibilityDescriptor},
      _blobType(blobType) {
    _version = METADATA_VERSION_2_7;
}

void MetadataBase::read(std::istream& stream) {
    _source = Source(stream);
    _sourceSize = get_stream_total_size(stream);
    read();

    // Note: we could have placed an additional safeguard here. Something like "cursorPosition = streamEnd -
    // footerSize", to make sure the whole content of the metadata section has been read. However, such a safeguard
    // would break compatibility, because some previous plugin versions are padding the space between the end of the
    // metadata and the footer.
}

void MetadataBase::read(const ov::Tensor& tensor) {
    _source = Source(tensor);
    _sourceSize = tensor.get_byte_size();
    read();
}

void MetadataBase::read_as_text(std::map<std::string, std::string, std::less<>> attrs) {
    _textAttrs = std::move(attrs);
    read_as_text();
}

size_t MetadataBase::get_remaining_source_size() const {
    size_t remaining;
    if (const std::reference_wrapper<std::istream>* stream =
            std::get_if<std::reference_wrapper<std::istream>>(&_source)) {
        OPENVINO_ASSERT(stream, STREAM_BAD_STATUS_MESSAGE);

        const auto offset = static_cast<size_t>(stream->get().tellg());
        remaining = (offset <= _sourceSize) ? _sourceSize - offset : 0;
    } else if (std::get_if<std::reference_wrapper<const ov::Tensor>>(&_source)) {
        remaining = (_cursorOffset <= _sourceSize) ? _sourceSize - _cursorOffset : 0;
    } else {
        OPENVINO_THROW(MISSING_BLOB_MESSAGE);
    }

    OPENVINO_ASSERT(remaining >= FOOTER_SIZE,
                    "Invalid state. While parsing the NPU plugin metadata, it was found that the remaining number of "
                    "bytes within the blob source is lower than the size of the footer.");
    return remaining;
}

void MetadataBase::read_data_from_source(char* destination, const size_t size) {
    const size_t remaining = get_remaining_source_size();
    OPENVINO_ASSERT(size <= remaining,
                    "NPU metadata: attempted to read ",
                    size,
                    " bytes but only ",
                    remaining,
                    " bytes remain in the metadata buffer.");

    if (const std::reference_wrapper<std::istream>* stream =
            std::get_if<std::reference_wrapper<std::istream>>(&_source)) {
        OPENVINO_ASSERT(stream, STREAM_BAD_STATUS_MESSAGE);

        stream->get().read(destination, size);
    } else if (const std::reference_wrapper<const ov::Tensor>* tensor =
                   std::get_if<std::reference_wrapper<const ov::Tensor>>(&_source)) {
        std::memcpy(destination, tensor->get().data<const char>() + _cursorOffset, size);
        _cursorOffset += size;
    } else {
        OPENVINO_THROW(MISSING_BLOB_MESSAGE);
    }
}

void MetadataBase::write(std::ostream& stream) {
    write_without_footer(stream);

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
        OPENVINO_ASSERT(
            numberOfInits <= (get_remaining_source_size() - FOOTER_SIZE) / SIZE_OF_INIT_SCHEDULE_SIZE,
            "The number of init schedules read from the blob is too great relative to the size of the blob");

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

    OPENVINO_ASSERT(numberOfInputLayouts + numberOfOutputLayouts <=
                        (get_remaining_source_size() - FOOTER_SIZE) / SIZE_OF_LAYOUT_SIZE,
                    "The number of I/O layouts read from the blob is too great relative to the size of the blob");

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
            OPENVINO_ASSERT(stringLength <= get_remaining_source_size() - FOOTER_SIZE,
                            "The size of at least one layout exceeds the limit of the blob");

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

void Metadata<METADATA_VERSION_2_6>::read() {
    Metadata<METADATA_VERSION_2_5>::read();

    uint64_t reqs_len;
    read_data_from_source(reinterpret_cast<char*>(&reqs_len), sizeof(reqs_len));
    if (reqs_len > 0) {
        OPENVINO_ASSERT(reqs_len <= (get_remaining_source_size() - FOOTER_SIZE),
                        "The size of the runtime requirements surpasses the limit of the blob");

        std::string reqs(reqs_len, '\0');
        read_data_from_source(reqs.data(), reqs_len);
        _compatibilityDescriptor = std::move(reqs);
    }
}

void Metadata<METADATA_VERSION_2_7>::read() {
    Metadata<METADATA_VERSION_2_6>::read();

    uint8_t blobType;
    read_data_from_source(reinterpret_cast<char*>(&blobType), sizeof(blobType));
    const auto type = static_cast<BlobType>(blobType);
    OPENVINO_ASSERT(type == BlobType::ELF || type == BlobType::LLVM || type == BlobType::BYTECODE,
                    "Invalid blob type in NPU blob metadata: ",
                    static_cast<uint32_t>(blobType));
    _blobType = type;
}

std::optional<BlobType> Metadata<METADATA_VERSION_2_7>::get_blob_type() const {
    return _blobType;
}

void Metadata<METADATA_VERSION_2_0>::read_as_text() {
    const auto it = _textAttrs.find(MetadataTextKeys::OV);
    if (it == _textAttrs.end()) {
        OPENVINO_THROW("Human-readable metadata missing '" + std::string(MetadataTextKeys::OV) + "' field.");
    }
    const auto ovParts = parse_version(it->second);
    if (ovParts.size() != 3) {
        OPENVINO_THROW("Human-readable metadata: '" + std::string(MetadataTextKeys::OV) +
                       "' is not in MAJOR.MINOR.PATCH format: " + it->second);
    }
    _ovVersion = OpenvinoVersion(ovParts[0], ovParts[1], ovParts[2]);
}

void Metadata<METADATA_VERSION_2_1>::read_as_text() {
    Metadata<METADATA_VERSION_2_0>::read_as_text();

    const auto it = _textAttrs.find(MetadataTextKeys::WS_INITS);
    if (it == _textAttrs.end()) {
        return;
    }
    if (it->second != "1") {
        OPENVINO_THROW("Human-readable metadata: '" + std::string(MetadataTextKeys::WS_INITS) +
                       "' must be '1' when present; got: " + it->second);
    }
    _initSizes = std::vector<uint64_t>{};
}

void Metadata<METADATA_VERSION_2_2>::read_as_text() {
    Metadata<METADATA_VERSION_2_1>::read_as_text();

    const auto it = _textAttrs.find(MetadataTextKeys::BATCH);
    if (it == _textAttrs.end()) {
        return;
    }
    const int64_t batchValue = std::stoll(it->second);
    _batchSize = batchValue != 0 ? std::optional<int64_t>(batchValue) : std::nullopt;
}

void Metadata<METADATA_VERSION_2_6>::read_as_text() {
    Metadata<METADATA_VERSION_2_5>::read_as_text();

    const auto it = _textAttrs.find(MetadataTextKeys::COMPAT_DESC);
    if (it == _textAttrs.end() || it->second.empty()) {
        return;
    }

    const std::string& v = it->second;
    if (v.size() >= 2 && v.front() == '[' && v.back() == ']') {
        _compatibilityDescriptor = v.substr(1, v.size() - 2);
    } else {
        OPENVINO_THROW("Human-readable metadata: 'desc' value is not bracket-enclosed: ", v);
    }
}

void Metadata<METADATA_VERSION_2_0>::write_without_footer(std::ostream& stream) {
    stream.write(reinterpret_cast<const char*>(&_version), sizeof(_version));
    _ovVersion.write(stream);
}

void Metadata<METADATA_VERSION_2_1>::write_without_footer(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_0>::write_without_footer(stream);

    _numberOfInits = _initSizes.has_value() ? _initSizes->size() : 0;
    stream.write(reinterpret_cast<const char*>(&_numberOfInits), sizeof(_numberOfInits));

    if (_initSizes.has_value()) {
        for (uint64_t initSize : _initSizes.value()) {
            stream.write(reinterpret_cast<const char*>(&initSize), sizeof(initSize));
        }
    }
}

void Metadata<METADATA_VERSION_2_2>::write_without_footer(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_1>::write_without_footer(stream);

    int64_t batchValue = _batchSize.value_or(0);
    stream.write(reinterpret_cast<const char*>(&batchValue), sizeof(batchValue));
}

void Metadata<METADATA_VERSION_2_3>::write_without_footer(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_2>::write_without_footer(stream);

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

void Metadata<METADATA_VERSION_2_4>::write_without_footer(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_3>::write_without_footer(stream);

    uint32_t compilerVersion = _compilerVersion.value_or(0);
    stream.write(reinterpret_cast<const char*>(&compilerVersion), sizeof(compilerVersion));
}

void Metadata<METADATA_VERSION_2_5>::write_without_footer(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_4>::write_without_footer(stream);

    const uint8_t isEncryptedBlob = _isEncryptedBlob.value_or(false);
    stream.write(reinterpret_cast<const char*>(&isEncryptedBlob), sizeof(isEncryptedBlob));
}

void Metadata<METADATA_VERSION_2_6>::write_without_footer(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_5>::write_without_footer(stream);

    const std::string& compatDesc = _compatibilityDescriptor.value_or("");
    const uint64_t compatDesc_len = compatDesc.size();
    stream.write(reinterpret_cast<const char*>(&compatDesc_len), sizeof(compatDesc_len));
    if (compatDesc_len > 0) {
        stream.write(compatDesc.data(), static_cast<std::streamsize>(compatDesc_len));
    }
}

void Metadata<METADATA_VERSION_2_7>::write_without_footer(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_6>::write_without_footer(stream);

    const auto blobType = static_cast<uint8_t>(_blobType);
    stream.write(reinterpret_cast<const char*>(&blobType), sizeof(blobType));
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
        write_text_field(stream, MetadataTextKeys::WS_INITS, "1");
    }
}

void Metadata<METADATA_VERSION_2_2>::write_as_text(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_1>::write_as_text(stream);

    if (_batchSize.has_value() && _batchSize.value() != 0) {
        write_text_field(stream, MetadataTextKeys::BATCH, _batchSize.value());
    }
}

void Metadata<METADATA_VERSION_2_6>::write_as_text(std::ostream& stream) {
    Metadata<METADATA_VERSION_2_5>::write_as_text(stream);

    if (_compatibilityDescriptor.has_value() && !_compatibilityDescriptor->empty()) {
        std::string desc = _compatibilityDescriptor.value();
        if (!desc.empty() && desc.back() == '\0') {
            desc.pop_back();
        }
        write_text_field(stream, MetadataTextKeys::COMPAT_DESC, '[' + desc + ']');
    }
}

std::unique_ptr<MetadataBase> create_metadata(uint32_t version, uint64_t blobSize) {
    auto logger = Logger::global().clone("create_metadata");

    switch (version) {
    case METADATA_VERSION_2_0:
        logger.debug("Creating a metadata object of version 2.0");
        return std::make_unique<Metadata<METADATA_VERSION_2_0>>(blobSize);
    case METADATA_VERSION_2_1:
        logger.debug("Creating a metadata object of version 2.1");
        return std::make_unique<Metadata<METADATA_VERSION_2_1>>(blobSize);
    case METADATA_VERSION_2_2:
        logger.debug("Creating a metadata object of version 2.2");
        return std::make_unique<Metadata<METADATA_VERSION_2_2>>(blobSize);
    case METADATA_VERSION_2_3:
        logger.debug("Creating a metadata object of version 2.3");
        return std::make_unique<Metadata<METADATA_VERSION_2_3>>(blobSize);
    case METADATA_VERSION_2_4:
        logger.debug("Creating a metadata object of version 2.4");
        return std::make_unique<Metadata<METADATA_VERSION_2_4>>(blobSize);
    case METADATA_VERSION_2_5:
        logger.debug("Creating a metadata object of version 2.5");
        return std::make_unique<Metadata<METADATA_VERSION_2_5>>(blobSize);
    case METADATA_VERSION_2_6:
        logger.debug("Creating a metadata object of version 2.6");
        return std::make_unique<Metadata<METADATA_VERSION_2_6>>(blobSize);
    case METADATA_VERSION_2_7:
        logger.debug("Creating a metadata object of version 2.7");
        return std::make_unique<Metadata<METADATA_VERSION_2_7>>(blobSize);
    default:
        OPENVINO_THROW("Metadata version is not supported! Imported blob metadata version: ",
                       MetadataBase::get_major(version),
                       ".",
                       MetadataBase::get_minor(version),
                       " but the current version is: ",
                       CURRENT_METADATA_MAJOR_VERSION,
                       ".",
                       CURRENT_METADATA_MINOR_VERSION);
    }
}

size_t MetadataBase::get_stream_remaining_size(std::istream& stream) {
    auto log = Logger::global().clone("get_stream_remaining_size");
    OPENVINO_ASSERT(stream, "Stream is in bad status! Please check the passed stream status!");

    if (dynamic_cast<ov::SharedStreamBuffer*>(stream.rdbuf()) != nullptr) {
        return stream.rdbuf()->in_avail();
    }
    const std::streampos streamStart = stream.tellg();
    stream.seekg(0, std::ios_base::end);
    const std::streampos streamEnd = stream.tellg();
    stream.seekg(streamStart, std::ios_base::beg);

    log.debug("Read blob size: streamStart=%zu, streamEnd=%zu",
              static_cast<size_t>(streamStart),
              static_cast<size_t>(streamEnd));

    OPENVINO_ASSERT(streamEnd >= streamStart,
                    "Invalid stream size: streamEnd (",
                    streamEnd,
                    ") is not larger than streamStart (",
                    streamStart,
                    ")!");

    return streamEnd - streamStart;
}

std::unique_ptr<MetadataBase> read_metadata_from(std::istream& stream) {
    std::streampos currentStreamPos = stream.tellg();
    const size_t streamSize = MetadataBase::get_stream_remaining_size(stream);

    OPENVINO_ASSERT(streamSize >= MINIMUM_BLOB_SIZE, BLOB_TOO_SMALL_MESSAGE, streamSize);

    std::string blobMagicBytes;
    blobMagicBytes.resize(MAGIC_BYTES.size());
    stream.seekg(-std::streampos(MAGIC_BYTES.size()), std::ios::end);
    stream.read(blobMagicBytes.data(), MAGIC_BYTES.size());
    OPENVINO_ASSERT(MAGIC_BYTES == blobMagicBytes, MISSING_METADATA_MESSAGE);

    uint64_t payloadSize;
    stream.seekg(-std::streampos(MAGIC_BYTES.size()) - sizeof(payloadSize), std::ios::end);
    stream.read(reinterpret_cast<char*>(&payloadSize), sizeof(payloadSize));

    OPENVINO_ASSERT(streamSize >= MINIMUM_BLOB_SIZE + payloadSize, INVALID_PAYLOAD_SIZE_MESSAGE, payloadSize);
    stream.seekg(-stream.tellg() + currentStreamPos + payloadSize, std::ios::cur);

    uint32_t metaVersion;
    stream.read(reinterpret_cast<char*>(&metaVersion), sizeof(metaVersion));

    std::unique_ptr<MetadataBase> storedMeta;
    try {
        storedMeta = create_metadata(metaVersion, payloadSize);
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
    const size_t blobSize = tensor.get_byte_size();
    OPENVINO_ASSERT(blobSize >= MINIMUM_BLOB_SIZE, BLOB_TOO_SMALL_MESSAGE, blobSize);

    const std::string_view blobMagicBytes(tensor.data<const char>() + blobSize - MAGIC_BYTES.size(),
                                          MAGIC_BYTES.size());
    OPENVINO_ASSERT(MAGIC_BYTES == blobMagicBytes, MISSING_METADATA_MESSAGE);

    uint64_t payloadSize;
    payloadSize = *reinterpret_cast<const decltype(payloadSize)*>(tensor.data<const char>() + blobSize -
                                                                  MAGIC_BYTES.size() - sizeof(payloadSize));

    OPENVINO_ASSERT(blobSize >= MINIMUM_BLOB_SIZE + payloadSize, INVALID_PAYLOAD_SIZE_MESSAGE, payloadSize);

    uint32_t metaVersion;
    metaVersion = *reinterpret_cast<const decltype(metaVersion)*>(tensor.data<const char>() + payloadSize);

    std::unique_ptr<MetadataBase> storedMeta;
    try {
        const ov::Tensor roiTensor(tensor, ov::Coordinate{payloadSize + sizeof(metaVersion)}, ov::Coordinate{blobSize});
        storedMeta = create_metadata(metaVersion, payloadSize);
        storedMeta->read(roiTensor);
    } catch (const std::exception& ex) {
        OPENVINO_THROW("Can't read NPU metadata: ", ex.what());
    } catch (...) {
        OPENVINO_THROW("Unexpected exception while reading blob NPU metadata");
    }

    return storedMeta;
}

std::unique_ptr<MetadataBase> read_as_text(std::string_view input) {
    std::string versionStr;
    compat::Parser::attr_map_type attrs;
    try {
        compat::Parser parser(input, metadataTextAttributes);
        versionStr = parser.getAttribute(std::string(MetadataTextKeys::META));
        attrs = parser.getAttributes();
    } catch (const std::exception& ex) {
        OPENVINO_THROW("NPU compatibility string is malformed: ", ex.what());
    }

    const auto metaParts = parse_version(versionStr);
    if (metaParts.size() != 2) {
        OPENVINO_THROW("NPU compatibility string is malformed: 'meta' must be in MAJOR.MINOR format: ", versionStr);
    }
    const uint32_t metaVersion = MetadataBase::make_version(metaParts[0], metaParts[1]);

    std::unique_ptr<MetadataBase> storedMeta;
    try {
        storedMeta = create_metadata(metaVersion, 0);
        storedMeta->read_as_text(std::move(attrs));
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

uint64_t MetadataBase::get_main_schedule_size() const {
    uint64_t accumulator = 0;
    const auto initSizes = get_init_sizes();
    return initSizes.has_value() ? get_blob_size() - std::accumulate(initSizes->begin(), initSizes->end(), accumulator)
                                 : get_blob_size();
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

std::optional<std::string_view> MetadataBase::get_compatibility_descriptor() const {
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

std::optional<std::string_view> Metadata<METADATA_VERSION_2_6>::get_compatibility_descriptor() const {
    return _compatibilityDescriptor;
}

}  // namespace intel_npu
