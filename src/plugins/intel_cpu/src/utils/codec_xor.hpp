// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <functional>
#include <string>

namespace ov {
namespace intel_cpu {

void codec_xor(char* dst_str, const char* src_str, size_t len);

std::string codec_xor_str(const std::string& source_str);

typedef std::function<std::string(const std::string&)>               CacheDecryptStr;
typedef std::function<void(char* dst, const char* src, size_t size)> CacheDecryptChar;

union CacheDecrypt {
    CacheDecryptChar m_decrypt_char = nullptr;
    CacheDecryptStr  m_decrypt_str;

    CacheDecrypt() {}

    CacheDecrypt(CacheDecryptStr fn)  : m_decrypt_str(fn) {}

    CacheDecrypt(CacheDecryptChar fn) : m_decrypt_char(fn) {}

    ~CacheDecrypt() {}

    operator bool() {
        return m_decrypt_char || m_decrypt_str;
    }
};

}  // namespace intel_cpu
}  // namespace ov
