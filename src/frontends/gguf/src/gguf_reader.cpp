// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gguf_reader.hpp"

#include <array>
#include <cstring>
#include <fstream>

#include "gguf_types.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"
#include "rt_info_keys.hpp"

namespace ov {
namespace frontend {
namespace gguf {

namespace {

constexpr uint32_t kGGUFMagic = 0x46554747;  // "GGUF" little-endian.
constexpr uint64_t kDefaultAlignment = 32;

// GGUF metadata value-type tags (little-endian on disk).
enum class ValueType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

// ---- Minimal self-contained SHA-256 (public-domain style) for source_file_hash ----
class Sha256 {
public:
    Sha256() {
        m_state = {0x6a09e667u,
                   0xbb67ae85u,
                   0x3c6ef372u,
                   0xa54ff53au,
                   0x510e527fu,
                   0x9b05688cu,
                   0x1f83d9abu,
                   0x5be0cd19u};
    }

    void update(const uint8_t* data, size_t len) {
        for (size_t i = 0; i < len; ++i) {
            m_buffer[m_buffer_len++] = data[i];
            if (m_buffer_len == 64) {
                transform(m_buffer.data());
                m_bit_len += 512;
                m_buffer_len = 0;
            }
        }
    }

    std::string hex_digest() {
        uint64_t total_bits = m_bit_len + static_cast<uint64_t>(m_buffer_len) * 8;
        size_t i = m_buffer_len;
        m_buffer[i++] = 0x80;
        if (i > 56) {
            while (i < 64)
                m_buffer[i++] = 0x00;
            transform(m_buffer.data());
            i = 0;
        }
        while (i < 56)
            m_buffer[i++] = 0x00;
        for (int b = 7; b >= 0; --b)
            m_buffer[i++] = static_cast<uint8_t>((total_bits >> (b * 8)) & 0xff);
        transform(m_buffer.data());

        static const char* hex = "0123456789abcdef";
        std::string out;
        out.reserve(64);
        for (uint32_t word : m_state) {
            for (int b = 3; b >= 0; --b) {
                uint8_t byte = static_cast<uint8_t>((word >> (b * 8)) & 0xff);
                out.push_back(hex[byte >> 4]);
                out.push_back(hex[byte & 0xf]);
            }
        }
        return out;
    }

private:
    static uint32_t rotr(uint32_t x, uint32_t n) {
        return (x >> n) | (x << (32 - n));
    }

    void transform(const uint8_t* chunk) {
        static const std::array<uint32_t, 64> k = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

        std::array<uint32_t, 64> w{};
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(chunk[i * 4]) << 24) | (static_cast<uint32_t>(chunk[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(chunk[i * 4 + 2]) << 8) | (static_cast<uint32_t>(chunk[i * 4 + 3]));
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = m_state[0], b = m_state[1], c = m_state[2], d = m_state[3];
        uint32_t e = m_state[4], f = m_state[5], g = m_state[6], h = m_state[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = h + s1 + ch + k[i] + w[i];
            uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        m_state[0] += a;
        m_state[1] += b;
        m_state[2] += c;
        m_state[3] += d;
        m_state[4] += e;
        m_state[5] += f;
        m_state[6] += g;
        m_state[7] += h;
    }

    std::array<uint32_t, 8> m_state{};
    std::array<uint8_t, 64> m_buffer{};
    size_t m_buffer_len = 0;
    uint64_t m_bit_len = 0;
};

// Bounds-checked little-endian cursor over the mapped file. All reads validate against the end
// of the buffer and fail fast with the absolute byte offset on truncation.
class Cursor {
public:
    Cursor(const uint8_t* base, size_t size) : m_base(base), m_size(size) {}

    template <class T>
    T read() {
        ensure(sizeof(T));
        T value;
        std::memcpy(&value, m_base + m_pos, sizeof(T));
        m_pos += sizeof(T);
        return value;
    }

    std::string read_string() {
        const uint64_t len = read<uint64_t>();
        ensure(len);
        std::string s(reinterpret_cast<const char*>(m_base + m_pos), static_cast<size_t>(len));
        m_pos += static_cast<size_t>(len);
        return s;
    }

    void ensure(uint64_t n) const {
        // m_pos <= m_size invariant holds, so (m_size - m_pos) does not underflow.
        OPENVINO_ASSERT(n <= m_size - m_pos,
                        "[GGUF Frontend] Truncated or malformed file: attempted to read ",
                        n,
                        " bytes at offset ",
                        m_pos,
                        " (file size ",
                        m_size,
                        ").");
    }

    size_t tell() const {
        return m_pos;
    }

private:
    const uint8_t* m_base;
    size_t m_size;
    size_t m_pos = 0;
};

// Reads a single scalar metadata value of the given type into an ov::Any.
ov::Any read_scalar_value(Cursor& cur, ValueType type) {
    switch (type) {
    case ValueType::UINT8:
        return ov::Any(cur.read<uint8_t>());
    case ValueType::INT8:
        return ov::Any(cur.read<int8_t>());
    case ValueType::UINT16:
        return ov::Any(cur.read<uint16_t>());
    case ValueType::INT16:
        return ov::Any(cur.read<int16_t>());
    case ValueType::UINT32:
        return ov::Any(cur.read<uint32_t>());
    case ValueType::INT32:
        return ov::Any(cur.read<int32_t>());
    case ValueType::FLOAT32:
        return ov::Any(cur.read<float>());
    case ValueType::BOOL:
        return ov::Any(static_cast<bool>(cur.read<uint8_t>()));
    case ValueType::STRING:
        return ov::Any(cur.read_string());
    case ValueType::UINT64:
        return ov::Any(cur.read<uint64_t>());
    case ValueType::INT64:
        return ov::Any(cur.read<int64_t>());
    case ValueType::FLOAT64:
        return ov::Any(cur.read<double>());
    default:
        OPENVINO_THROW("[GGUF Frontend] Unexpected metadata value type id: ", static_cast<uint32_t>(type));
    }
}

template <class T>
ov::Any read_scalar_array(Cursor& cur, uint64_t count, ValueType elem) {
    std::vector<T> values;
    values.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i)
        values.push_back(read_scalar_value(cur, elem).as<T>());
    return ov::Any(std::move(values));
}

ov::Any read_value(Cursor& cur, ValueType type) {
    if (type != ValueType::ARRAY)
        return read_scalar_value(cur, type);

    const auto elem = static_cast<ValueType>(cur.read<uint32_t>());
    const uint64_t count = cur.read<uint64_t>();
    switch (elem) {
    case ValueType::UINT8:
        return read_scalar_array<uint8_t>(cur, count, elem);
    case ValueType::INT8:
        return read_scalar_array<int8_t>(cur, count, elem);
    case ValueType::UINT16:
        return read_scalar_array<uint16_t>(cur, count, elem);
    case ValueType::INT16:
        return read_scalar_array<int16_t>(cur, count, elem);
    case ValueType::UINT32:
        return read_scalar_array<uint32_t>(cur, count, elem);
    case ValueType::INT32:
        return read_scalar_array<int32_t>(cur, count, elem);
    case ValueType::FLOAT32:
        return read_scalar_array<float>(cur, count, elem);
    case ValueType::BOOL:
        return read_scalar_array<bool>(cur, count, elem);
    case ValueType::STRING:
        return read_scalar_array<std::string>(cur, count, elem);
    case ValueType::UINT64:
        return read_scalar_array<uint64_t>(cur, count, elem);
    case ValueType::INT64:
        return read_scalar_array<int64_t>(cur, count, elem);
    case ValueType::FLOAT64:
        return read_scalar_array<double>(cur, count, elem);
    default:
        OPENVINO_THROW("[GGUF Frontend] Unsupported array element type id: ", static_cast<uint32_t>(elem));
    }
}

bool any_to_u64(const ov::Any& a, uint64_t& out) {
    if (a.is<uint8_t>()) {
        out = a.as<uint8_t>();
    } else if (a.is<int8_t>()) {
        out = static_cast<uint64_t>(a.as<int8_t>());
    } else if (a.is<uint16_t>()) {
        out = a.as<uint16_t>();
    } else if (a.is<int16_t>()) {
        out = static_cast<uint64_t>(a.as<int16_t>());
    } else if (a.is<uint32_t>()) {
        out = a.as<uint32_t>();
    } else if (a.is<int32_t>()) {
        out = static_cast<uint64_t>(a.as<int32_t>());
    } else if (a.is<uint64_t>()) {
        out = a.as<uint64_t>();
    } else if (a.is<int64_t>()) {
        out = static_cast<uint64_t>(a.as<int64_t>());
    } else if (a.is<bool>()) {
        out = a.as<bool>() ? 1u : 0u;
    } else {
        return false;
    }
    return true;
}

bool any_to_f64(const ov::Any& a, double& out) {
    if (a.is<float>()) {
        out = a.as<float>();
        return true;
    }
    if (a.is<double>()) {
        out = a.as<double>();
        return true;
    }
    uint64_t as_int = 0;
    if (any_to_u64(a, as_int)) {
        out = static_cast<double>(as_int);
        return true;
    }
    return false;
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
    if (alignment == 0)
        alignment = 1;
    return ((value + alignment - 1) / alignment) * alignment;
}

size_t physical_byte_size(const ov::element::Type& type, size_t num_elements) {
    if (type.is_gguf_block()) {
        const size_t bec = type.block_elem_count();
        const size_t bbs = type.block_byte_size();
        OPENVINO_ASSERT(bec != 0, "[GGUF Frontend] Invalid block element count for ", type.get_type_name());
        return ((num_elements + bec - 1) / bec) * bbs;
    }
    // Remaining GGUF tensor types (F32/F16/BF16/I8/I16/I32/I64/F64) are byte-aligned.
    return num_elements * type.size();
}

}  // namespace

GGUFReader::GGUFReader(const std::string& path, bool mmap_enable) : m_mmap_enable(mmap_enable) {
    if (m_mmap_enable) {
        m_mmap = ov::load_mmap_object(path);
        OPENVINO_ASSERT(m_mmap, "[GGUF Frontend] Failed to memory-map '", path, "'.");
        m_base = reinterpret_cast<const uint8_t*>(m_mmap->data());
        m_total_size = m_mmap->size();
    } else {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        OPENVINO_ASSERT(stream.is_open(), "[GGUF Frontend] Failed to open '", path, "'.");
        const std::streamsize size = stream.tellg();
        OPENVINO_ASSERT(size > 0, "[GGUF Frontend] Empty or unreadable file '", path, "'.");
        stream.seekg(0, std::ios::beg);
        m_buffer = std::make_shared<ov::AlignedBuffer>(static_cast<size_t>(size));
        OPENVINO_ASSERT(stream.read(m_buffer->get_ptr<char>(), size),
                        "[GGUF Frontend] Failed to read '",
                        path,
                        "'.");
        m_base = m_buffer->get_ptr<uint8_t>();
        m_total_size = static_cast<size_t>(size);
    }
    parse();
}

void GGUFReader::parse() {
    Cursor cur(m_base, m_total_size);

    const uint32_t magic = cur.read<uint32_t>();
    OPENVINO_ASSERT(magic == kGGUFMagic,
                    "[GGUF Frontend] Not a GGUF file (bad magic 0x",
                    std::hex,
                    magic,
                    ").");
    m_version = cur.read<uint32_t>();
    OPENVINO_ASSERT(m_version == 2 || m_version == 3,
                    "[GGUF Frontend] Unsupported GGUF version: ",
                    m_version,
                    " (supported: 2, 3).");

    const uint64_t tensor_count = cur.read<uint64_t>();
    const uint64_t metadata_count = cur.read<uint64_t>();

    for (uint64_t i = 0; i < metadata_count; ++i) {
        std::string key = cur.read_string();
        const auto value_type = static_cast<ValueType>(cur.read<uint32_t>());
        m_metadata.emplace(std::move(key), read_value(cur, value_type));
    }

    m_architecture = get_str(file_keys::general_architecture, std::string{});
    const uint64_t alignment = get_u64(file_keys::general_alignment, kDefaultAlignment);

    m_tensors.reserve(static_cast<size_t>(tensor_count));
    for (uint64_t i = 0; i < tensor_count; ++i) {
        GGUFTensorInfo info;
        info.name = cur.read_string();
        const uint32_t n_dims = cur.read<uint32_t>();
        OPENVINO_ASSERT(n_dims <= 4,
                        "[GGUF Frontend] Tensor '",
                        info.name,
                        "' has unsupported rank ",
                        n_dims,
                        ".");

        std::vector<uint64_t> ggml_dims(n_dims);
        size_t num_elements = 1;
        for (uint32_t d = 0; d < n_dims; ++d) {
            ggml_dims[d] = cur.read<uint64_t>();
            num_elements *= static_cast<size_t>(ggml_dims[d]);
        }
        // GGML stores dims fastest-first; OpenVINO logical shape is output-dim-first => reverse.
        info.shape = ov::Shape(ggml_dims.rbegin(), ggml_dims.rend());

        info.ggml_type = cur.read<uint32_t>();
        info.type = ggml_type_to_ov_element_type(info.ggml_type);
        info.data_offset = cur.read<uint64_t>();
        info.byte_size = physical_byte_size(info.type, num_elements);

        m_tensor_index.emplace(info.name, m_tensors.size());
        m_tensors.push_back(std::move(info));
    }

    m_data_start = align_up(cur.tell(), alignment);
    OPENVINO_ASSERT(m_data_start <= m_total_size,
                    "[GGUF Frontend] Data section start (",
                    m_data_start,
                    ") is beyond end of file (",
                    m_total_size,
                    ").");

    // Validate every tensor lies fully within the file (untrusted offsets/sizes).
    for (const auto& info : m_tensors) {
        const uint64_t end = m_data_start + info.data_offset + info.byte_size;
        OPENVINO_ASSERT(info.data_offset <= m_total_size - m_data_start && end <= m_total_size,
                        "[GGUF Frontend] Tensor '",
                        info.name,
                        "' data (offset ",
                        info.data_offset,
                        ", size ",
                        info.byte_size,
                        ") exceeds file bounds.");
    }

    Sha256 hasher;
    hasher.update(m_base, m_total_size);
    m_file_hash = hasher.hex_digest();
}

const GGUFTensorInfo* GGUFReader::find_tensor(const std::string& name) const {
    const auto it = m_tensor_index.find(name);
    return it == m_tensor_index.end() ? nullptr : &m_tensors[it->second];
}

std::shared_ptr<ov::op::v0::Constant> GGUFReader::make_constant(const ov::element::Type& type,
                                                                const ov::Shape& shape,
                                                                const uint8_t* ptr,
                                                                size_t byte_size) const {
    if (m_mmap_enable) {
        auto shared = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
            const_cast<char*>(reinterpret_cast<const char*>(ptr)),
            byte_size,
            m_mmap);
        return std::make_shared<ov::op::v0::Constant>(type, shape, shared);
    }
    auto shared = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
        const_cast<char*>(reinterpret_cast<const char*>(ptr)),
        byte_size,
        m_buffer);
    return std::make_shared<ov::op::v0::Constant>(type, shape, shared);
}

std::shared_ptr<ov::op::v0::Constant> GGUFReader::tensor_constant(const std::string& name) const {
    const GGUFTensorInfo* info = find_tensor(name);
    OPENVINO_ASSERT(info, "[GGUF Frontend] Tensor '", name, "' not found.");
    const uint8_t* ptr = m_base + m_data_start + info->data_offset;
    return make_constant(info->type, info->shape, ptr, info->byte_size);
}

const uint8_t* GGUFReader::tensor_data(const std::string& name, size_t& byte_size) const {
    const GGUFTensorInfo* info = find_tensor(name);
    OPENVINO_ASSERT(info, "[GGUF Frontend] Tensor '", name, "' not found.");
    byte_size = info->byte_size;
    return m_base + m_data_start + info->data_offset;
}

bool GGUFReader::has(const std::string& key) const {
    return m_metadata.find(key) != m_metadata.end();
}

const ov::Any& GGUFReader::raw(const std::string& key) const {
    const auto it = m_metadata.find(key);
    OPENVINO_ASSERT(it != m_metadata.end(), "[GGUF Frontend] Required metadata key '", key, "' is missing.");
    return it->second;
}

uint64_t GGUFReader::get_u64(const std::string& key) const {
    uint64_t out = 0;
    OPENVINO_ASSERT(any_to_u64(raw(key), out), "[GGUF Frontend] Metadata key '", key, "' is not an integer.");
    return out;
}

uint64_t GGUFReader::get_u64(const std::string& key, uint64_t default_value) const {
    uint64_t out = 0;
    if (has(key) && any_to_u64(raw(key), out))
        return out;
    return default_value;
}

double GGUFReader::get_f64(const std::string& key) const {
    double out = 0;
    OPENVINO_ASSERT(any_to_f64(raw(key), out), "[GGUF Frontend] Metadata key '", key, "' is not a number.");
    return out;
}

double GGUFReader::get_f64(const std::string& key, double default_value) const {
    double out = 0;
    if (has(key) && any_to_f64(raw(key), out))
        return out;
    return default_value;
}

std::string GGUFReader::get_str(const std::string& key) const {
    const ov::Any& value = raw(key);
    OPENVINO_ASSERT(value.is<std::string>(), "[GGUF Frontend] Metadata key '", key, "' is not a string.");
    return value.as<std::string>();
}

std::string GGUFReader::get_str(const std::string& key, const std::string& default_value) const {
    if (has(key)) {
        const ov::Any& value = raw(key);
        if (value.is<std::string>())
            return value.as<std::string>();
    }
    return default_value;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
