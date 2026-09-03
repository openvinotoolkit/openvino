// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// A minimal GGUF writer, for tests that need a file no fixture provides.
//
// The per-architecture fixtures come from llama.cpp's `test-llama-archs`, which only emits causal
// decoders. A non-decoder family -- an mmproj vision encoder -- therefore has no fixture at all,
// and the extension mechanism's whole point is that such a family can be added from outside, so
// there has to be a way to produce one here. This writes the header the parser reads (magic, KV
// metadata, tensor table) and zero-fills the tensor data, which is all graph construction depends
// on -- the same reasoning that lets test_arch_conversion.cpp rebuild its fixtures from headers.

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace ov_gguf_test {

// GGUF metadata value types (gguf.h GGUF_VALUE_TYPE_*).
enum : uint32_t {
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
};

// ggml tensor type F32 (ggml.h GGML_TYPE_F32).
constexpr uint32_t GGML_TYPE_F32 = 0;

class GgufWriter {
public:
    void kv_u32(const std::string& key, uint32_t v) {
        kv_key(key, GGUF_TYPE_UINT32);
        put(v);
    }
    void kv_f32(const std::string& key, float v) {
        kv_key(key, GGUF_TYPE_FLOAT32);
        put(v);
    }
    void kv_bool(const std::string& key, bool v) {
        kv_key(key, GGUF_TYPE_BOOL);
        m_kv.push_back(static_cast<char>(v ? 1 : 0));
    }
    void kv_str(const std::string& key, const std::string& v) {
        kv_key(key, GGUF_TYPE_STRING);
        put(static_cast<uint64_t>(v.size()));
        m_kv.insert(m_kv.end(), v.begin(), v.end());
    }

    // Declare a tensor. `dims` is in GGUF on-disk order (fastest-varying first), as the format
    // stores it -- the reverse of the OpenVINO shape it becomes.
    void tensor(const std::string& name, const std::vector<uint64_t>& dims) {
        put_str(m_ti, name);
        put(m_ti, static_cast<uint32_t>(dims.size()));
        for (auto d : dims) {
            put(m_ti, d);
        }
        put(m_ti, GGML_TYPE_F32);
        put(m_ti, m_data_size);

        uint64_t n = 1;
        for (auto d : dims) {
            n *= d;
        }
        uint64_t bytes = n * sizeof(float);
        // Every tensor starts at an aligned offset within the data section.
        if (const uint64_t rem = bytes % kAlignment) {
            bytes += kAlignment - rem;
        }
        m_data_size += bytes;
        ++m_n_tensors;
    }

    bool write(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            return false;
        }
        std::vector<char> head;
        put(head, static_cast<uint32_t>(0x46554747));  // "GGUF"
        put(head, static_cast<uint32_t>(3));           // version
        put(head, m_n_tensors);
        put(head, m_n_kv);
        out.write(head.data(), static_cast<std::streamsize>(head.size()));
        out.write(m_kv.data(), static_cast<std::streamsize>(m_kv.size()));
        out.write(m_ti.data(), static_cast<std::streamsize>(m_ti.size()));

        // Tensor data starts at the next aligned offset after the info section.
        uint64_t off = head.size() + m_kv.size() + m_ti.size();
        if (const uint64_t rem = off % kAlignment) {
            const std::vector<char> pad(kAlignment - rem, 0);
            out.write(pad.data(), static_cast<std::streamsize>(pad.size()));
        }
        const std::vector<char> zeros(64 * 1024, 0);
        uint64_t remaining = m_data_size;
        while (remaining > 0) {
            const uint64_t chunk = std::min<uint64_t>(remaining, zeros.size());
            out.write(zeros.data(), static_cast<std::streamsize>(chunk));
            remaining -= chunk;
        }
        return static_cast<bool>(out);
    }

private:
    static constexpr uint64_t kAlignment = 32;

    template <typename T>
    static void put(std::vector<char>& buf, const T& v) {
        const auto* p = reinterpret_cast<const char*>(&v);
        buf.insert(buf.end(), p, p + sizeof(T));
    }
    static void put_str(std::vector<char>& buf, const std::string& s) {
        put(buf, static_cast<uint64_t>(s.size()));
        buf.insert(buf.end(), s.begin(), s.end());
    }
    template <typename T>
    void put(const T& v) {
        put(m_kv, v);
    }
    void kv_key(const std::string& key, uint32_t type) {
        put_str(m_kv, key);
        put(m_kv, type);
        ++m_n_kv;
    }

    std::vector<char> m_kv;
    std::vector<char> m_ti;
    uint64_t m_n_kv = 0;
    uint64_t m_n_tensors = 0;
    uint64_t m_data_size = 0;
};

}  // namespace ov_gguf_test
