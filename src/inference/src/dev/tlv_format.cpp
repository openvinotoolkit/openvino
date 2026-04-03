// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/tlv_format.hpp"

#include <type_traits>
#include <utility>

namespace ov::runtime {

void write_tlv_record(std::ostream& stream, TLVTraits::TagType tag, TLVTraits::LengthType size, const char* data) {
    stream.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
    stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
    if (size && data) {
        stream.write(reinterpret_cast<const char*>(data), size);
    }
}

void write_tlv_record(std::ostream& stream, TLVTraits::TagType tag, const TLVValueWriter& writer) {
    stream.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
    const auto size_pos = stream.tellp();
    TLVTraits::LengthType size = 0;
    stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
    if (writer) {
        const auto value_pos = stream.tellp();
        writer(stream);
        const auto end_pos = stream.tellp();
        size = end_pos - value_pos;
        stream.seekp(size_pos);
        stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
        stream.seekp(end_pos);
    }
}

namespace {
template <typename Container>
static bool read_record(std::istream& stream, TLVTraits::TagType& tag, TLVTraits::LengthType& size, Container& data) {
    stream.read(reinterpret_cast<char*>(&tag), sizeof(tag));
    if (!stream.good()) {
        return false;
    }
    stream.read(reinterpret_cast<char*>(&size), sizeof(size));
    if (!stream.good()) {
        return false;
    }
    if (size == 0) {
        data.clear();
        return true;
    }
    const auto current_pos = stream.tellg();
    const auto remaining_offset = stream.seekg(0, std::ios::end).tellg() - current_pos;
    stream.seekg(current_pos);
    if (!stream.good() || remaining_offset < static_cast<std::streamoff>(size)) {
        return false;
    }
    data.resize(size);
    stream.read(reinterpret_cast<char*>(data.data()), size);
    return stream.good();
}
}  // namespace

bool read_tlv_record(std::istream& stream,
                     TLVTraits::TagType& tag,
                     TLVTraits::LengthType& size,
                     std::vector<char>& data) {
    return read_record(stream, tag, size, data);
}

bool scan_tlv_records(std::istream& stream, const TLVValueScanner& scanners) {
    const auto beginning_pos = stream.tellg();
    stream.seekg(0, std::ios::end);
    const auto stream_end = stream.tellg();
    stream.seekg(beginning_pos);

    while (stream.good() && stream.tellg() < stream_end) {
        TLVTraits::TagType tag{};
        TLVTraits::LengthType size{};
        stream.read(reinterpret_cast<char*>(&tag), sizeof(tag));
        if (!stream.good()) {
            return false;
        }
        stream.read(reinterpret_cast<char*>(&size), sizeof(size));
        if (!stream.good()) {
            return false;
        }

        if (auto scanner_it = scanners.find(tag); scanner_it != scanners.end()) {
            if (!scanner_it->second(stream, size)) {
                return false;
            }
        } else {
            if (stream_end - stream.tellg() < static_cast<std::streamoff>(size) || !stream.good()) {
                return false;
            }
            stream.seekg(size, std::ios::cur);
            if (!stream.good()) {
                return false;
            }
        }
    }
    return true;
}
}  // namespace ov::runtime
