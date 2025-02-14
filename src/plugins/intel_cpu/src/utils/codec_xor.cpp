// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/codec_xor.hpp"

#include "openvino/core/parallel.hpp"

namespace ov::intel_cpu {

void codec_xor(char* dst_str, const char* src_str, size_t len) {
    static const char codec_key[] = {0x30, 0x60, 0x70, 0x02, 0x04, 0x08, 0x3F, 0x6F, 0x72, 0x74, 0x78, 0x7F};
    auto key_size = sizeof(codec_key);

    if (dst_str == src_str) {
        parallel_for(len, [&](size_t key_idx) {
            dst_str[key_idx] ^= codec_key[key_idx % key_size];
        });
    } else {
        parallel_for(len, [&](size_t key_idx) {
            dst_str[key_idx] = src_str[key_idx] ^ codec_key[key_idx % key_size];
        });
    }
}

std::string codec_xor_str(const std::string& source_str) {
    std::string new_str(source_str);
    codec_xor(&new_str[0], &new_str[0], new_str.size());
    return new_str;
}

}  // namespace ov::intel_cpu
   // namespace ov.
