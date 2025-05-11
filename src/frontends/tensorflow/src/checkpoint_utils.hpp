// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdint.h>

#include <fstream>
#include <vector>

#include "openvino/frontend/exception.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

#define VARIABLES_INDEX_FOOTER_SIZE 48
#define BLOCK_TRAILER_SIZE          5
#define SAVED_TENSOR_SLICES_KEY     ""
#define META_GRAPH_DEFAULT_TAG      "serve"

template <typename T>
static T smUnpack(char*& ptr, const char* ptr_end) {
    T result = 0;
    for (uint8_t i = 0; i <= sizeof(T) * 7 && ptr < ptr_end; i += 7) {
        T byte = *(ptr++);
        if (byte & 0x80) {
            result |= ((byte & 0x7F) << i);
        } else {
            result |= byte << i;
            return result;
        }
    }
    return 0;
}

/// \brief Structure is for storing information about block in Varaibles Index file.
/// It defines only offset and block size, no information about exact content.
struct VIBlock {
    uint64_t m_size;
    uint64_t m_offset;

    void read(char*& ptr, const char* ptr_end) {
        m_offset = smUnpack<uint64_t>(ptr, ptr_end);
        m_size = smUnpack<uint64_t>(ptr, ptr_end);
    }
};

/// \brief Structure is for storing information about Variables Index footer information.
/// It contains description of two blocks and a magic number for a file verification.
/// Currently, it is placed in last VARIABLES_INDEX_FOOTER_SIZE bytes at the end of a file.
struct VIFooter {
    VIBlock m_metaIndex;
    VIBlock m_index;

    void read(char*& ptr, const char* ptr_end) {
        m_index.read(ptr, ptr_end);
        m_metaIndex.read(ptr, ptr_end);
    }

    void read(std::ifstream& fs) {
        fs.seekg(0, std::ios::end);
        size_t size = fs.tellg();
        FRONT_END_GENERAL_CHECK(size >= VARIABLES_INDEX_FOOTER_SIZE,
                                "Wrong index file, file size is less than minimal expected");

        char footerData[VARIABLES_INDEX_FOOTER_SIZE] = {}, *ptr = &footerData[0];
        fs.seekg(size - sizeof(footerData));
        fs.read(ptr, sizeof(footerData));

        // https://github.com/tensorflow/tensorflow/blob/9659b7bdca80a8ef8240eb021d4da089034eeb00/tensorflow/tsl/lib/io/format.cc#L59
        ptr += sizeof(footerData) - 8;
        uint32_t magic_lo = *reinterpret_cast<const uint32_t*>(ptr);
        uint32_t magic_hi = *reinterpret_cast<const uint32_t*>(ptr + 4);
        uint64_t magic_no = (static_cast<uint64_t>(magic_hi) << 32) | static_cast<uint64_t>(magic_lo);

        FRONT_END_GENERAL_CHECK(magic_no == 0xdb4775248b80fb57ull, "Wrong index file, magic number mismatch detected");

        ptr = &footerData[0];
        m_metaIndex.read(ptr, ptr + sizeof(footerData));
        m_index.read(ptr, ptr + sizeof(footerData));
    }
};

uint32_t decode_fixed32(const char* ptr);

const char* decode_entry(const char* p,
                         const char* limit,
                         uint32_t& shared,
                         uint32_t& non_shared,
                         uint32_t& value_length);

bool get_varint64(std::string& input, uint64_t* value);

std::string encode_tensor_name_slice(const std::string& name,
                                     const std::vector<int64_t>& starts,
                                     const std::vector<int64_t> lengths);
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov