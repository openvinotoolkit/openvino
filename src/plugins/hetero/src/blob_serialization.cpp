// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_serialization.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <sstream>

#include "openvino/core/except.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/icore.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::hetero {
namespace {

std::streamsize checked_stream_size(std::uint64_t size, const char* fieldName) {
    OPENVINO_ASSERT(size <= static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()),
                    "HETERO compiled blob ",
                    fieldName,
                    " size is too large: ",
                    size);
    return static_cast<std::streamsize>(size);
}

std::streamoff checked_stream_offset(std::uint64_t offset, const char* fieldName) {
    OPENVINO_ASSERT(offset <= static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()),
                    "HETERO compiled blob ",
                    fieldName,
                    " offset is too large: ",
                    offset);
    return static_cast<std::streamoff>(offset);
}

void write_bytes(std::ostream& stream, const char* data, std::uint64_t size, const char* fieldName) {
    stream.write(data, checked_stream_size(size, fieldName));
    OPENVINO_ASSERT(stream, "Failed to write HETERO compiled blob ", fieldName);
}

std::uint64_t read_size(std::istream& stream, const char* fieldName) {
    std::uint64_t size = 0;
    read_payload_bytes(stream, reinterpret_cast<char*>(&size), sizeof(size), fieldName);
    return size;
}

void consume_payload_bytes(std::uint64_t& remainingPayloadSize, std::uint64_t size, const char* fieldName) {
    if (remainingPayloadSize == std::numeric_limits<std::uint64_t>::max()) {
        return;
    }

    OPENVINO_ASSERT(size <= remainingPayloadSize,
                    "HETERO compiled blob ",
                    fieldName,
                    " size exceeds remaining IR payload: ",
                    size,
                    " > ",
                    remainingPayloadSize);
    remainingPayloadSize -= size;
}

std::uint64_t remaining_stream_bytes(std::istream& stream) {
    const auto currentPos = stream.tellg();
    if (currentPos == std::streampos(-1)) {
        return std::numeric_limits<std::uint64_t>::max();
    }

    stream.clear();
    stream.seekg(0, std::ios_base::end);
    const auto endPos = stream.tellg();
    stream.clear();
    stream.seekg(currentPos);

    if (endPos == std::streampos(-1) || endPos < currentPos || !stream) {
        stream.clear();
        return std::numeric_limits<std::uint64_t>::max();
    }

    return static_cast<std::uint64_t>(endPos - currentPos);
}

void write_size(std::ostream& stream, std::uint64_t size, const char* fieldName) {
    stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
    OPENVINO_ASSERT(stream, "Failed to write HETERO compiled blob ", fieldName);
}

}  // namespace

BoundedStreamBuffer::BoundedStreamBuffer(std::istream& stream, std::uint64_t size)
    : _stream(stream),
      _start(stream.tellg()),
      _size(size) {
    _seekable = _start != std::streampos(-1);
    checked_stream_offset(_size, "payload");
}

std::streampos BoundedStreamBuffer::end_pos() const {
    OPENVINO_ASSERT(_seekable, "HETERO compiled blob input stream is not seekable");
    return _start + checked_stream_offset(_size, "payload");
}

void BoundedStreamBuffer::consume_remaining_payload() {
    std::uint64_t remaining = _size - _pos;
    if (remaining == 0) {
        return;
    }

    if (_has_current) {
        _has_current = false;
        ++_pos;
        --remaining;
    }

    std::array<char, 4096> scratch{};
    while (remaining > 0) {
        const auto chunk = static_cast<std::streamsize>(
            std::min<std::uint64_t>(remaining, static_cast<std::uint64_t>(scratch.size())));
        _stream.read(scratch.data(), chunk);
        const auto read = _stream.gcount();
        OPENVINO_ASSERT(read == chunk && !_stream.bad(),
                        "Failed to consume remaining HETERO compiled blob payload bytes");
        _pos += static_cast<std::uint64_t>(read);
        remaining -= static_cast<std::uint64_t>(read);
    }
}

std::streamsize BoundedStreamBuffer::xsgetn(char* data, std::streamsize count) {
    if (count <= 0 || _pos >= _size) {
        return 0;
    }

    std::streamsize totalRead = 0;
    if (_has_current) {
        *data = _current;
        _has_current = false;
        ++_pos;
        ++data;
        --count;
        ++totalRead;
    }

    if (count <= 0 || _pos >= _size) {
        return totalRead;
    }

    const auto bytesToRead = std::min(static_cast<std::uint64_t>(count), _size - _pos);
    _stream.read(data, static_cast<std::streamsize>(bytesToRead));

    const auto bytesRead = _stream.gcount();
    _pos += static_cast<std::uint64_t>(bytesRead);
    totalRead += bytesRead;
    return totalRead;
}

BoundedStreamBuffer::int_type BoundedStreamBuffer::uflow() {
    if (_has_current) {
        _has_current = false;
        ++_pos;
        return traits_type::to_int_type(_current);
    }

    if (_pos >= _size) {
        return traits_type::eof();
    }

    _stream.read(&_current, 1);
    if (_stream.gcount() != 1) {
        return traits_type::eof();
    }

    ++_pos;
    return traits_type::to_int_type(_current);
}

BoundedStreamBuffer::int_type BoundedStreamBuffer::underflow() {
    if (_has_current) {
        return traits_type::to_int_type(_current);
    }

    if (_pos >= _size) {
        return traits_type::eof();
    }

    _stream.read(&_current, 1);
    if (_stream.gcount() != 1) {
        return traits_type::eof();
    }

    _has_current = true;
    return traits_type::to_int_type(_current);
}

std::streamsize BoundedStreamBuffer::showmanyc() {
    const auto available = _size - _pos;
    return static_cast<std::streamsize>(
        std::min<std::uint64_t>(available, static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())));
}

BoundedStreamBuffer::pos_type BoundedStreamBuffer::seekoff(off_type off,
                                                           std::ios_base::seekdir dir,
                                                           std::ios_base::openmode which) {
    if ((which & std::ios_base::in) == 0) {
        return pos_type(off_type(-1));
    }

    if (!_seekable) {
        if (dir == std::ios_base::cur && off == 0) {
            return pos_type(static_cast<off_type>(_pos));
        }
        return pos_type(off_type(-1));
    }

    off_type base = 0;
    if (dir == std::ios_base::cur) {
        base = static_cast<off_type>(_pos);
    } else if (dir == std::ios_base::end) {
        base = static_cast<off_type>(_size);
    } else if (dir != std::ios_base::beg) {
        return pos_type(off_type(-1));
    }

    const auto newPos = base + off;
    if (newPos < 0 || static_cast<std::uint64_t>(newPos) > _size) {
        return pos_type(off_type(-1));
    }

    _pos = static_cast<std::uint64_t>(newPos);
    _has_current = false;
    _stream.clear();
    _stream.seekg(_start + checked_stream_offset(_pos, "payload position"));
    if (!_stream) {
        return pos_type(off_type(-1));
    }
    setg(nullptr, nullptr, nullptr);
    return pos_type(static_cast<off_type>(_pos));
}

BoundedStreamBuffer::pos_type BoundedStreamBuffer::seekpos(pos_type pos, std::ios_base::openmode which) {
    return seekoff(static_cast<off_type>(pos), std::ios_base::beg, which);
}

void read_payload_bytes(std::istream& stream, char* data, std::uint64_t size, const char* fieldName) {
    const auto streamSize = checked_stream_size(size, fieldName);
    if (streamSize == 0) {
        return;
    }
    stream.read(data, streamSize);
    OPENVINO_ASSERT(stream.gcount() == streamSize && !stream.bad(), "Failed to read HETERO compiled blob ", fieldName);
}

FramedPayloadOutputBuffer::FramedPayloadOutputBuffer(std::ostream& stream)
    : _stream(stream),
      _start(stream.tellp()),
      _underlyingPos(_start) {
    OPENVINO_ASSERT(_start != std::streampos(-1), "HETERO compiled blob output stream is not seekable");
}

std::uint64_t FramedPayloadOutputBuffer::written_size() const {
    return _writtenSize;
}

std::streamsize FramedPayloadOutputBuffer::xsputn(const char* data, std::streamsize count) {
    if (count <= 0) {
        return 0;
    }

    const auto newPos = _pos + static_cast<std::uint64_t>(count);
    checked_stream_offset(newPos, "payload position");

    _stream.clear();
    const auto writePos = _start + checked_stream_offset(_pos, "payload position");
    if (_underlyingPos != writePos) {
        _stream.seekp(writePos);
        _underlyingPos = writePos;
    }
    _stream.write(data, count);
    OPENVINO_ASSERT(_stream, "Failed to write HETERO compiled blob payload data");

    _pos = newPos;
    _underlyingPos = _start + checked_stream_offset(_pos, "payload position");
    _writtenSize = std::max(_writtenSize, _pos);
    return count;
}

FramedPayloadOutputBuffer::int_type FramedPayloadOutputBuffer::overflow(int_type ch) {
    if (traits_type::eq_int_type(ch, traits_type::eof())) {
        return traits_type::not_eof(ch);
    }

    const char c = traits_type::to_char_type(ch);
    return xsputn(&c, 1) == 1 ? ch : traits_type::eof();
}

FramedPayloadOutputBuffer::pos_type FramedPayloadOutputBuffer::seekoff(off_type off,
                                                                       std::ios_base::seekdir dir,
                                                                       std::ios_base::openmode which) {
    if ((which & std::ios_base::out) == 0) {
        return pos_type(off_type(-1));
    }

    off_type base = 0;
    if (dir == std::ios_base::cur) {
        base = static_cast<off_type>(_pos);
    } else if (dir == std::ios_base::end) {
        base = static_cast<off_type>(_writtenSize);
    } else if (dir != std::ios_base::beg) {
        return pos_type(off_type(-1));
    }

    const auto newPos = base + off;
    if (newPos < 0) {
        return pos_type(off_type(-1));
    }

    if (static_cast<std::uint64_t>(newPos) > _writtenSize) {
        return pos_type(off_type(-1));
    }
    const auto uNewPos = static_cast<std::uint64_t>(newPos);
    checked_stream_offset(uNewPos, "payload position");
    _pos = uNewPos;
    return pos_type(static_cast<off_type>(_pos));
}

FramedPayloadOutputBuffer::pos_type FramedPayloadOutputBuffer::seekpos(pos_type pos, std::ios_base::openmode which) {
    return seekoff(static_cast<off_type>(pos), std::ios_base::beg, which);
}

BoundedStringOutputBuffer::BoundedStringOutputBuffer(std::uint64_t maxSize) : _maxSize(maxSize) {
    checked_stream_size(maxSize, "buffered payload");
}

const std::string& BoundedStringOutputBuffer::str() const {
    return _data;
}

std::streamsize BoundedStringOutputBuffer::xsputn(const char* data, std::streamsize count) {
    if (count <= 0) {
        return 0;
    }

    const auto writeSize = static_cast<std::uint64_t>(count);
    OPENVINO_ASSERT(writeSize <= _maxSize - static_cast<std::uint64_t>(_data.size()),
                    "HETERO compiled blob buffered payload size exceeds limit: ",
                    static_cast<std::uint64_t>(_data.size()) + writeSize,
                    " > ",
                    _maxSize);

    _data.append(data, static_cast<std::string::size_type>(count));
    return count;
}

BoundedStringOutputBuffer::int_type BoundedStringOutputBuffer::overflow(int_type ch) {
    if (traits_type::eq_int_type(ch, traits_type::eof())) {
        return traits_type::not_eof(ch);
    }

    const char c = traits_type::to_char_type(ch);
    return xsputn(&c, 1) == 1 ? ch : traits_type::eof();
}

bool is_output_stream_seekable(std::ostream& model_stream) {
    const auto pos = model_stream.tellp();
    if (pos == std::streampos(-1)) {
        return false;
    }

    const auto state = model_stream.rdstate();
    model_stream.clear();
    model_stream.seekp(pos);
    const bool seekable = static_cast<bool>(model_stream);
    model_stream.clear(state);
    return seekable;
}

PayloadFrame start_framed_payload(std::ostream& model_stream, char payloadType) {
    const auto frameStartPos = model_stream.tellp();
    OPENVINO_ASSERT(frameStartPos != std::streampos(-1), "HETERO compiled blob output stream is not seekable");

    write_bytes(model_stream, &payloadType, sizeof(payloadType), "payload type");
    const auto sizePos = model_stream.tellp();
    OPENVINO_ASSERT(sizePos != std::streampos(-1), "HETERO compiled blob output stream is not seekable");

    write_size(model_stream, 0, "payload size");
    const auto payloadStartPos = model_stream.tellp();
    OPENVINO_ASSERT(payloadStartPos != std::streampos(-1), "HETERO compiled blob output stream is not seekable");

    return {frameStartPos, sizePos, payloadStartPos};
}

void finish_framed_payload(std::ostream& model_stream, const PayloadFrame& payloadFrame) {
    const auto payloadEndPos = model_stream.tellp();
    OPENVINO_ASSERT(payloadEndPos != std::streampos(-1), "HETERO compiled blob output stream is not seekable");
    OPENVINO_ASSERT(payloadEndPos >= payloadFrame.payload_start_pos,
                    "HETERO compiled blob payload end position is before payload start");

    const auto payloadSize = static_cast<std::uint64_t>(payloadEndPos - payloadFrame.payload_start_pos);

    finish_framed_payload(model_stream, payloadFrame, payloadSize);
}

void finish_framed_payload(std::ostream& model_stream, const PayloadFrame& payloadFrame, std::uint64_t payloadSize) {
    const auto payloadEndPos = payloadFrame.payload_start_pos + checked_stream_offset(payloadSize, "payload");

    model_stream.clear();
    model_stream.seekp(payloadFrame.size_pos);
    write_size(model_stream, payloadSize, "payload size");

    model_stream.clear();
    model_stream.seekp(payloadEndPos);
    OPENVINO_ASSERT(model_stream, "Failed to restore HETERO compiled blob output stream position");
}

void write_framed_payload(std::ostream& model_stream, char payloadType, const std::string& payload) {
    write_bytes(model_stream, &payloadType, sizeof(payloadType), "payload type");
    write_size(model_stream, static_cast<std::uint64_t>(payload.size()), "payload size");
    write_bytes(model_stream, payload.data(), payload.size(), "payload data");
}

void write_framed_payload(std::ostream& model_stream,
                          char payloadType,
                          std::istream& payloadStream,
                          std::uint64_t payloadSize) {
    write_bytes(model_stream, &payloadType, sizeof(payloadType), "payload type");
    write_size(model_stream, payloadSize, "payload size");

    std::array<char, 4096> buffer{};
    std::uint64_t remaining = payloadSize;
    while (remaining > 0) {
        const auto chunk = std::min<std::uint64_t>(remaining, static_cast<std::uint64_t>(buffer.size()));
        read_payload_bytes(payloadStream, buffer.data(), chunk, "payload data");
        write_bytes(model_stream, buffer.data(), chunk, "payload data");
        remaining -= chunk;
    }
}

PayloadHeader read_payload_header(std::istream& model_stream) {
    PayloadHeader payloadHeader;
    read_payload_bytes(model_stream, &payloadHeader.type, sizeof(payloadHeader.type), "payload type");
    payloadHeader.size = read_size(model_stream, "payload size");
    return payloadHeader;
}

void read_ir_payload(std::istream& model,
                     const std::shared_ptr<ov::ICore>& core,
                     const std::string& device,
                     const ov::AnyMap& loadConfig,
                     std::shared_ptr<ov::Model>& ov_model,
                     ov::SoPtr<ov::ICompiledModel>& compiled_model,
                     std::uint64_t payloadSize) {
    const auto hasPayloadBoundary = payloadSize != std::numeric_limits<std::uint64_t>::max();
    auto remainingPayloadSize = hasPayloadBoundary ? payloadSize : remaining_stream_bytes(model);

    std::string xmlString;
    consume_payload_bytes(remainingPayloadSize, sizeof(std::uint64_t), "IR XML size");
    auto dataSize = read_size(model, "IR XML size");
    consume_payload_bytes(remainingPayloadSize, dataSize, "IR XML content");
    OPENVINO_ASSERT(dataSize <= static_cast<std::uint64_t>(std::numeric_limits<std::string::size_type>::max()),
                    "HETERO compiled blob IR XML size is too large: ",
                    dataSize);
    xmlString.resize(static_cast<std::string::size_type>(dataSize));
    read_payload_bytes(model, xmlString.data(), dataSize, "IR XML content");

    ov::Tensor weights;
    consume_payload_bytes(remainingPayloadSize, sizeof(std::uint64_t), "IR weights size");
    dataSize = read_size(model, "IR weights size");
    consume_payload_bytes(remainingPayloadSize, dataSize, "IR weights content");
    if (dataSize != 0) {
        OPENVINO_ASSERT(dataSize <= static_cast<std::uint64_t>(std::numeric_limits<ov::Shape::size_type>::max()),
                        "HETERO compiled blob IR weights size is too large: ",
                        dataSize);
        weights = ov::Tensor(ov::element::u8, ov::Shape{static_cast<ov::Shape::size_type>(dataSize)});
        read_payload_bytes(model,
                           reinterpret_cast<char*>(weights.data<std::uint8_t>()),
                           dataSize,
                           "IR weights content");
    }

    OPENVINO_ASSERT(!hasPayloadBoundary || remainingPayloadSize == 0,
                    "HETERO compiled blob IR payload has unexpected trailing data: ",
                    remainingPayloadSize);

    ov_model = core->read_model(xmlString, weights);
    compiled_model = core->compile_model(ov_model, device, loadConfig);
}

void write_ir_payload(std::ostream& model_stream, const std::shared_ptr<ov::Model>& model) {
    if (!model) {
        OPENVINO_THROW("OpenVINO Model is empty");
    }

    std::stringstream xmlFile, binFile;
    ov::pass::Serialize serializer(xmlFile, binFile);
    serializer.run_on_model(model);

    auto constants = binFile.str();
    auto model_str = xmlFile.str();

    auto dataSize = static_cast<std::uint64_t>(model_str.size());
    write_size(model_stream, dataSize, "IR XML size");
    write_bytes(model_stream, model_str.data(), dataSize, "IR XML content");

    dataSize = static_cast<std::uint64_t>(constants.size());
    write_size(model_stream, dataSize, "IR weights size");
    write_bytes(model_stream, constants.data(), dataSize, "IR weights content");
}

}  // namespace ov::hetero
