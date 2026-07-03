// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_serialization.hpp"

#include <algorithm>
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

void read_bytes(std::istream& stream, char* data, std::uint64_t size, const char* fieldName) {
    const auto streamSize = checked_stream_size(size, fieldName);
    if (streamSize == 0) {
        return;
    }
    stream.read(data, streamSize);
    OPENVINO_ASSERT(stream.gcount() == streamSize, "Failed to read HETERO compiled blob ", fieldName);
}

void write_bytes(std::ostream& stream, const char* data, std::uint64_t size, const char* fieldName) {
    stream.write(data, checked_stream_size(size, fieldName));
}

std::uint64_t read_size(std::istream& stream, const char* fieldName) {
    std::uint64_t size = 0;
    read_bytes(stream, reinterpret_cast<char*>(&size), sizeof(size), fieldName);
    return size;
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
    OPENVINO_ASSERT(_start != std::streampos(-1), "HETERO compiled blob input stream is not seekable");
}

std::streampos BoundedStreamBuffer::end_pos() const {
    return _start + static_cast<std::streamoff>(_size);
}

std::streamsize BoundedStreamBuffer::xsgetn(char* data, std::streamsize count) {
    if (count <= 0 || _pos >= _size) {
        return 0;
    }

    const auto bytesToRead = std::min(static_cast<std::uint64_t>(count), _size - _pos);
    _stream.clear();
    _stream.seekg(_start + static_cast<std::streamoff>(_pos));
    _stream.read(data, static_cast<std::streamsize>(bytesToRead));

    const auto bytesRead = _stream.gcount();
    _pos += static_cast<std::uint64_t>(bytesRead);
    return bytesRead;
}

BoundedStreamBuffer::int_type BoundedStreamBuffer::uflow() {
    char c = 0;
    return xsgetn(&c, 1) == 1 ? traits_type::to_int_type(c) : traits_type::eof();
}

BoundedStreamBuffer::int_type BoundedStreamBuffer::underflow() {
    if (_pos >= _size) {
        return traits_type::eof();
    }

    _stream.clear();
    _stream.seekg(_start + static_cast<std::streamoff>(_pos));
    _stream.read(&_current, 1);
    return _stream.gcount() == 1 ? traits_type::to_int_type(_current) : traits_type::eof();
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
    setg(nullptr, nullptr, nullptr);
    return pos_type(static_cast<off_type>(_pos));
}

BoundedStreamBuffer::pos_type BoundedStreamBuffer::seekpos(pos_type pos, std::ios_base::openmode which) {
    return seekoff(static_cast<off_type>(pos), std::ios_base::beg, which);
}

FramedPayloadOutputBuffer::FramedPayloadOutputBuffer(std::ostream& stream)
    : _stream(stream),
      _start(stream.tellp()) {
    OPENVINO_ASSERT(_start != std::streampos(-1), "HETERO compiled blob output stream is not seekable");
}

std::uint64_t FramedPayloadOutputBuffer::written_size() const {
    return _writtenSize;
}

std::streamsize FramedPayloadOutputBuffer::xsputn(const char* data, std::streamsize count) {
    if (count <= 0) {
        return 0;
    }

    _stream.clear();
    _stream.seekp(_start + static_cast<std::streamoff>(_pos));
    _stream.write(data, count);
    _pos += static_cast<std::uint64_t>(count);
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

    _pos = static_cast<std::uint64_t>(newPos);
    return pos_type(static_cast<off_type>(_pos));
}

FramedPayloadOutputBuffer::pos_type FramedPayloadOutputBuffer::seekpos(pos_type pos, std::ios_base::openmode which) {
    return seekoff(static_cast<off_type>(pos), std::ios_base::beg, which);
}

PayloadFrame start_framed_payload(std::ostream& model_stream, char payloadType) {
    const auto typePos = model_stream.tellp();
    OPENVINO_ASSERT(typePos != std::streampos(-1), "HETERO compiled blob output stream is not seekable");
    model_stream.write(&payloadType, sizeof(payloadType));
    const auto sizePos = model_stream.tellp();
    OPENVINO_ASSERT(sizePos != std::streampos(-1), "HETERO compiled blob output stream is not seekable");
    std::uint64_t payloadSize = 0;
    model_stream.write(reinterpret_cast<char*>(&payloadSize), sizeof(payloadSize));
    const auto payloadStartPos = model_stream.tellp();
    OPENVINO_ASSERT(payloadStartPos != std::streampos(-1), "HETERO compiled blob output stream is not seekable");
    return {typePos, sizePos, payloadStartPos};
}

void finish_framed_payload(std::ostream& model_stream, const PayloadFrame& payloadFrame, std::uint64_t payloadSize) {
    const auto payloadEndPos = payloadFrame.payload_start_pos + static_cast<std::streamoff>(payloadSize);
    model_stream.seekp(payloadFrame.size_pos);
    write_size(model_stream, payloadSize, "payload size");
    model_stream.seekp(payloadEndPos);
}

std::uint64_t finish_framed_payload(std::ostream& model_stream, const PayloadFrame& payloadFrame) {
    const auto payloadEndPos = model_stream.tellp();
    OPENVINO_ASSERT(payloadEndPos != std::streampos(-1), "HETERO compiled blob output stream is not seekable");
    const auto payloadSize = static_cast<std::uint64_t>(payloadEndPos - payloadFrame.payload_start_pos);
    finish_framed_payload(model_stream, payloadFrame, payloadSize);
    return payloadSize;
}

PayloadHeader read_payload_header(std::istream& model_stream) {
    PayloadHeader payloadHeader;
    read_bytes(model_stream, &payloadHeader.type, sizeof(payloadHeader.type), "payload type");
    payloadHeader.size = read_size(model_stream, "payload size");
    return payloadHeader;
}

void read_ir_payload(std::istream& model,
                     const std::shared_ptr<ov::ICore>& core,
                     const std::string& device,
                     const ov::AnyMap& loadConfig,
                     std::shared_ptr<ov::Model>& ov_model,
                     ov::SoPtr<ov::ICompiledModel>& compiled_model) {
    std::string xmlString;
    auto dataSize = read_size(model, "IR XML size");
    xmlString.resize(dataSize);
    read_bytes(model, xmlString.data(), dataSize, "IR XML content");

    ov::Tensor weights;
    dataSize = read_size(model, "IR weights size");
    if (dataSize != 0) {
        weights = ov::Tensor(ov::element::u8, ov::Shape{static_cast<ov::Shape::size_type>(dataSize)});
        read_bytes(model, weights.data<char>(), dataSize, "IR weights content");
    }

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
