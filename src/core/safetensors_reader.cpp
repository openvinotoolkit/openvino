// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "safetensors_reader.hpp"

#include "runtime/file_util.hpp"

#include <cstdint>
#include <cstring>
#include <map>
#include <string_view>

namespace ov::core {
namespace vulkan {
namespace cross_platform {
namespace st_r {

namespace {

[[noreturn]] void st_error(const std::string& msg) {
    throw std::runtime_error("[GPU] safetensors: " + msg);
}

// ---- minimal JSON scanner for the safetensors header schema ----------------

class json_scanner {
public:
    explicit json_scanner(std::string_view s) : _s(s) {}

    void skip_ws() {
        while (_pos < _s.size() && (_s[_pos] == ' ' || _s[_pos] == '\t' || _s[_pos] == '\n' || _s[_pos] == '\r'))
            ++_pos;
    }
    bool eof() { skip_ws(); return _pos >= _s.size(); }
    char peek() {
        skip_ws();
        if (_pos >= _s.size())
            st_error("unexpected end of JSON header");
        return _s[_pos];
    }
    void expect(char c) {
        if (peek() != c)
            st_error(std::string("expected '") + c + "' at offset " + std::to_string(_pos));
        ++_pos;
    }
    bool try_consume(char c) {
        if (_pos < _s.size() && _s[_pos] == c) {
            ++_pos;
            return true;
        }
        return false;
    }

    std::string read_string() {
        expect('"');
        std::string out;
        while (_pos < _s.size() && _s[_pos] != '"') {
            if (_s[_pos] == '\\' && _pos + 1 < _s.size()) {
                ++_pos;  // keep escapes simple: drop the backslash
                out.push_back(_s[_pos++]);
            } else {
                out.push_back(_s[_pos++]);
            }
        }
        if (_pos >= _s.size())
            st_error("unterminated string");
        ++_pos;  // closing quote
        return out;
    }

    uint64_t read_u64() {
        skip_ws();
        const size_t start = _pos;
        while (_pos < _s.size() && _s[_pos] >= '0' && _s[_pos] <= '9')
            ++_pos;
        if (_pos == start)
            st_error("expected number at offset " + std::to_string(start));
        return std::stoull(std::string(_s.substr(start, _pos - start)));
    }

    // Reads a [a, b, ...] of unsigned integers.
    std::vector<uint64_t> read_u64_array() {
        expect('[');
        std::vector<uint64_t> out;
        if (try_consume(']'))
            return out;
        while (true) {
            out.push_back(read_u64());
            if (try_consume(','))
                continue;
            expect(']');
            break;
        }
        return out;
    }

private:
    std::string_view _s;
    size_t _pos = 0;
};

float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        bits = man == 0 ? sign : sign | ((127 - 15 + 1) << 23);
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

}  // namespace

std::map<std::string, st_tensor> load_safetensors(const std::string& path) {
    auto raw_bytes = ov::util::load_binary(path);  // vector<std::byte>
    std::string_view blob{reinterpret_cast<const char*>(raw_bytes.data()), raw_bytes.size()};
    if (blob.size() < 8)
        st_error("file too small for the header length: " + path);

    uint64_t hdr_len = 0;
    std::memcpy(&hdr_len, blob.data(), 8);
    if (8 + hdr_len > blob.size())
        st_error("header length " + std::to_string(hdr_len) + " exceeds file size");

    json_scanner js(blob.substr(8, hdr_len));
    const std::string_view data_block = blob.substr(8 + hdr_len);

    std::map<std::string, st_tensor> out;
    js.expect('{');
    if (js.eof())
        st_error("empty JSON header");
    while (true) {
        if (js.try_consume('}'))
            break;
        const auto name = js.read_string();
        js.expect(':');
        if (name == "__metadata__") {
            // Skip an arbitrary object value.
            int depth = 0;
            while (true) {
                const char c = js.peek();
                if (c == '{') {
                    ++depth;
                    js.expect('{');
                } else if (c == '}') {
                    js.expect('}');
                    if (--depth == 0)
                        break;
                } else if (c == '"') {
                    (void)js.read_string();
                } else if (c == ',') {
                    js.expect(',');
                } else if (c == ':') {
                    js.expect(':');
                } else {
                    js.expect(c);
                }
            }
            if (js.try_consume(','))
                continue;
            js.expect('}');
            break;
        }
        // Tensor descriptor object.
        js.expect('{');
        std::string dtype;
        std::vector<uint64_t> shape, offs;
        while (true) {
            const auto key = js.read_string();
            js.expect(':');
            if (key == "dtype")
                dtype = js.read_string();
            else if (key == "shape")
                shape = js.read_u64_array();
            else if (key == "data_offsets")
                offs = js.read_u64_array();
            else if (js.peek() == '"')
                (void)js.read_string();
            else
                (void)js.read_u64_array();
            if (js.try_consume(','))
                continue;
            js.expect('}');
            break;
        }
        if (offs.size() != 2 || offs[1] < offs[0])
            st_error("tensor '" + name + "': bad data_offsets");
        if (dtype != "F32" && dtype != "F16" && dtype != "BF16")
            st_error("tensor '" + name + "': unsupported dtype " + dtype +
                     " (supported: F32, F16, BF16)");
        if (offs[1] > data_block.size())
            st_error("tensor '" + name + "': data range exceeds the data block");

        const size_t elem = static_cast<size_t>(offs[1] - offs[0]);
        st_tensor t;
        t.shape.reserve(shape.size());
        size_t count = 1;
        for (const auto d : shape) {
            t.shape.push_back(static_cast<size_t>(d));
            count *= static_cast<size_t>(d);
        }
        const size_t width = dtype == "F32" ? 4u : 2u;
        if (count * width != elem)
            st_error("tensor '" + name + "': byte size does not match the shape");
        t.data.resize(count);
        const uint8_t* src = reinterpret_cast<const uint8_t*>(data_block.data()) + offs[0];
        if (dtype == "F32") {
            std::memcpy(t.data.data(), src, elem);
        } else if (dtype == "F16") {
            for (size_t i = 0; i < count; ++i)
                t.data[i] = f16_to_f32(static_cast<uint16_t>(src[2 * i] | (src[2 * i + 1] << 8)));
        } else {  // BF16: high half of an f32.
            for (size_t i = 0; i < count; ++i) {
                const uint32_t bits = static_cast<uint32_t>(src[2 * i]) << 16 |
                                      static_cast<uint32_t>(src[2 * i + 1]) << 24;
                std::memcpy(&t.data[i], &bits, 4);
            }
        }
        out.emplace(std::move(name), std::move(t));

        if (js.try_consume(','))
            continue;
        js.expect('}');
        break;
    }
    return out;
}

}  // namespace st_r
}  // namespace cross_platform
}  // namespace vulkan
}  // namespace ov
