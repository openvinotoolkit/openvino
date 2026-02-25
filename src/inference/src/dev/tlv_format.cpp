// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/tlv_format.hpp"

#include <type_traits>
#include <utility>

namespace ov::runtime {

// todo General: How to handle stream not good?

void TLVFormat::write_entry(std::ostream& stream, TagType tag, LenghtType size, const uint8_t* data) {
    stream.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
    stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
    if (size && data) {
        stream.write(reinterpret_cast<const char*>(data), size);
    }
}

void TLVFormat::write_entry(std::ostream& stream, TagType tag, const ValueWriter& writer) {
    stream.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
    const auto size_pos = stream.tellp();
    LenghtType size = 0;
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

template <typename Container>
static bool read_entry_(std::istream& stream, TLVFormat::TagType& tag, TLVFormat::LenghtType& size, Container& data) {
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
    data.resize(size);
    stream.read(reinterpret_cast<char*>(data.data()), size);
    return stream.good();
}

bool TLVFormat::read_entry(std::istream& stream, TagType& tag, LenghtType& size, std::vector<uint8_t>& data) {
    return read_entry_(stream, tag, size, data);
}
bool TLVFormat::read_entry(std::istream& stream, TagType& tag, LenghtType& size, std::string& data) {
    return read_entry_(stream, tag, size, data);
}

void TLVFormat::scan_entries(std::istream& stream, const ValueScanner& scanners, bool rewind) {
    const auto beginning_pos = stream.tellg();
    stream.seekg(0, std::ios::end);
    const auto stream_end = stream.tellg();
    stream.seekg(beginning_pos);

    while (stream.good() && stream.tellg() < stream_end) {
        TLVFormat::TagType tag{};
        TLVFormat::LenghtType size{};
        stream.read(reinterpret_cast<char*>(&tag), sizeof(tag));
        if (!stream.good()) {
            break;
        }
        stream.read(reinterpret_cast<char*>(&size), sizeof(size));
        if (!stream.good()) {
            break;
        }

        if (auto scanner_it = scanners.find(tag); scanner_it != scanners.end()) {
            scanner_it->second(stream, size);
        } else {
            stream.seekg(size, std::ios::cur);
        }
    }
    if (rewind) {
        stream.seekg(beginning_pos, std::ios::beg);
    }
}
}  // namespace ov::runtime
