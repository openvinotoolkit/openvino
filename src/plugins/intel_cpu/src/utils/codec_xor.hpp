// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <utility>

namespace ov::intel_cpu {

void codec_xor(char* dst_str, const char* src_str, size_t len);

std::string codec_xor_str(const std::string& source_str);

using CacheDecryptStr = std::function<std::string(const std::string&)>;
using CacheDecryptChar = std::function<void(char*, const char*, size_t)>;

union CacheDecrypt {
    CacheDecryptChar m_decrypt_char = nullptr;
    CacheDecryptStr m_decrypt_str;

    CacheDecrypt() {}

    explicit CacheDecrypt(CacheDecryptStr fn) : m_decrypt_str(std::move(fn)) {}

    explicit CacheDecrypt(CacheDecryptChar fn) : m_decrypt_char(std::move(fn)) {}

    ~CacheDecrypt() {}

    explicit operator bool() const {
        return m_decrypt_char || m_decrypt_str;
    }
};

}  // namespace ov::intel_cpu
