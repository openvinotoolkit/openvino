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
void finish_framed_payload(std::ostream& model_stream, const PayloadFrame& payloadFrame);
void write_framed_payload(std::ostream& model_stream, char payloadType, const std::string& payload);
PayloadHeader read_payload_header(std::istream& model_stream);

void read_ir_payload(std::istream& model,
                     const std::shared_ptr<ov::ICore>& core,
                     const std::string& device,
                     const ov::AnyMap& loadConfig,
                     std::shared_ptr<ov::Model>& ov_model,
                     ov::SoPtr<ov::ICompiledModel>& compiled_model,
                     std::uint64_t payloadSize);

void write_ir_payload(std::ostream& model_stream, const std::shared_ptr<ov::Model>& model);

}  // namespace ov::hetero
