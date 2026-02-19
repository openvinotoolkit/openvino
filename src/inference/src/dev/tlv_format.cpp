// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/tlv_format.hpp"

#include <type_traits>
#include <utility>

// template <typename Container, typename = void>
// struct has_required_non_static_members : std::false_type {};

// template <typename Container>
// struct has_required_non_static_members<
//     Container,
//     std::void_t<typename Container::size_type,
//                 typename Container::value_type,
//                 decltype(static_cast<void (Container::*)()>(&Container::clear)),
//                 decltype(static_cast<void (Container::*)(typename Container::size_type)>(&Container::resize)),
//                 decltype(static_cast<typename Container::value_type* (Container::*)()>(&Container::data))>>
//     : std::true_type {};

namespace ov {

// todo General: How to handle stream not good?

void TLVFormat::write_entry(std::ostream& stream, tag_type tag, length_type size, const uint8_t* data) {
    stream.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
    stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
    if (size && data) {
        stream.write(reinterpret_cast<const char*>(data), size);
    }
}

void TLVFormat::write_entry(std::ostream& stream, tag_type tag, const value_writer_callable& writer) {
    stream.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
    const auto size_pos = stream.tellp();
    length_type size = 0;
    stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
    const auto value_pos = stream.tellp();
    writer(stream);
    const auto end_pos = stream.tellp();
    size = end_pos - value_pos;
    stream.seekp(size_pos);
    stream.write(reinterpret_cast<const char*>(&size), sizeof(size));
    stream.seekp(end_pos);
}

template <typename Container>
static bool read_entry_(std::istream& stream, TLVFormat::tag_type& tag, TLVFormat::length_type& size, Container& data) {
    // static_assert(has_required_non_static_members<Container>::value,
    //               "Container type must define non-static clear(), resize(size_type), and data() member functions");
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

bool TLVFormat::read_entry(std::istream& stream, tag_type& tag, length_type& size, std::vector<uint8_t>& data) {
    return read_entry_(stream, tag, size, data);
}
bool TLVFormat::read_entry(std::istream& stream, tag_type& tag, length_type& size, std::string& data) {
    return read_entry_(stream, tag, size, data);
}

void TLVFormat::scan_entries(std::istream& stream, const value_scanners& scanners, bool rewind) {
    const auto beginning_pos = stream.tellg();
    stream.seekg(0, std::ios::end);
    const auto stream_end = stream.tellg();
    stream.seekg(beginning_pos);

    while (stream.good() && stream.tellg() < stream_end) {
        TLVFormat::tag_type tag{};
        TLVFormat::length_type size{};
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
}  // namespace ov
