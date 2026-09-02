// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <ios>
#include <istream>
#include <memory>
#include <ostream>
#include <streambuf>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
class ICore;
class ICompiledModel;
class Model;
}  // namespace ov

namespace ov::hetero {

// HETERO compiled blob layout (blob_format_version = 2, framed format):
//
//   [blob header XML]
//     One UTF-8 XML line written by pugi::xml_document::save(..., format_raw)
//     and terminated by '\n'.
//     This XML stores the HETERO metadata (model name, format version,
//     I/O mapping, hetero config, and compiled_submodel entries).
//
//   | blob header XML | submodel payload frame #0 | submodel payload frame #1 | ... | submodel payload frame #N-1 |
//     The XML header comes first and lists the compiled_submodel entries.
//     After that, one payload frame is written for each entry, in the same order.
//
// Submodel payload frame (repeated once per compiled_submodel in the XML header order):
//
//   +--------------------+--------------------------+----------------------+
//   | type (1 byte char) | payload size (uint64_t)  | payload raw data     |
//   +--------------------+--------------------------+----------------------+
//
// Payload type values:
//   'B' (COMPILED_BLOB_PAYLOAD): payload raw data is the backend-specific
//     compiled blob for one submodel.
//   'I' (IR_PAYLOAD): payload raw data contains a serialized IR fallback payload.
//     Its internal layout is:
//
//       +----------------------+------------------+-------------------------+--------------------+
//       | IR XML size (u64)    | IR XML raw data  | IR weights size (u64)   | IR weights raw data|
//       +----------------------+------------------+-------------------------+--------------------+
//
// Notes:
//   - uint64_t values are written/read as raw host-endian bytes.
//   - blob_format_version = 1 is a legacy unframed format handled by the reader
//     for backward compatibility.

constexpr std::uint32_t HETERO_BLOB_FORMAT_VERSION = 2;
constexpr const char* HETERO_BLOB_FORMAT_VERSION_ATTR = "blob_format_version";
constexpr char COMPILED_BLOB_PAYLOAD = 'B';
constexpr char IR_PAYLOAD = 'I';
constexpr std::uint64_t MAX_IN_MEMORY_COMPILED_PAYLOAD_SIZE = 64ULL * 1024ULL * 1024ULL;

class BoundedStreamBuffer : public std::streambuf {
public:
    BoundedStreamBuffer(std::istream& stream, std::uint64_t size);

    std::streampos end_pos() const;
    void consume_remaining_payload();

protected:
    std::streamsize xsgetn(char* data, std::streamsize count) override;
    int_type uflow() override;
    int_type underflow() override;
    std::streamsize showmanyc() override;
    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override;
    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override;

private:
    std::istream& _stream;
    std::streampos _start;
    bool _seekable = false;
    std::uint64_t _size = 0;
    std::uint64_t _pos = 0;
    char _current = 0;
    bool _has_current = false;
};

class FramedPayloadOutputBuffer : public std::streambuf {
public:
    explicit FramedPayloadOutputBuffer(std::ostream& stream);

    std::uint64_t written_size() const;

protected:
    std::streamsize xsputn(const char* data, std::streamsize count) override;
    int_type overflow(int_type ch) override;
    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override;
    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override;

private:
    std::ostream& _stream;
    std::streampos _start;
    std::streampos _underlyingPos;
    std::uint64_t _pos = 0;
    std::uint64_t _writtenSize = 0;
};

class BoundedStringOutputBuffer : public std::streambuf {
public:
    explicit BoundedStringOutputBuffer(std::uint64_t maxSize);

    const std::string& str() const;

protected:
    std::streamsize xsputn(const char* data, std::streamsize count) override;
    int_type overflow(int_type ch) override;

private:
    std::uint64_t _maxSize = 0;
    std::string _data;
};

struct PayloadHeader {
    char type = 0;
    std::uint64_t size = 0;
};

struct PayloadFrame {
    std::streampos frame_start_pos;
    std::streampos size_pos;
    std::streampos payload_start_pos;
};

bool is_output_stream_seekable(std::ostream& model_stream);
PayloadFrame start_framed_payload(std::ostream& model_stream, char payloadType);
void finish_framed_payload(std::ostream& model_stream, const PayloadFrame& payloadFrame, std::uint64_t payloadSize);
void finish_framed_payload(std::ostream& model_stream, const PayloadFrame& payloadFrame);
void write_framed_payload(std::ostream& model_stream, char payloadType, const std::string& payload);
void write_framed_payload(std::ostream& model_stream,
                          char payloadType,
                          std::istream& payloadStream,
                          std::uint64_t payloadSize);
PayloadHeader read_payload_header(std::istream& model_stream);
void read_payload_bytes(std::istream& stream, char* data, std::uint64_t size, const char* fieldName);

void read_ir_payload(std::istream& model,
                     const std::shared_ptr<ov::ICore>& core,
                     const std::string& device,
                     const ov::AnyMap& loadConfig,
                     std::shared_ptr<ov::Model>& ov_model,
                     ov::SoPtr<ov::ICompiledModel>& compiled_model,
                     std::uint64_t payloadSize);

void write_ir_payload(std::ostream& model_stream, const std::shared_ptr<ov::Model>& model);

}  // namespace ov::hetero
