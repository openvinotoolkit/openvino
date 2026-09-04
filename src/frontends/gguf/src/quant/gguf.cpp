// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Native GGUF container parser. Replaces the third-party gguflib dependency: the file is
// memory-mapped via ov::load_mmap_object and parsed directly per the GGUF v2/v3 format
// (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md). No llama.cpp / ggml
// dependency.

#include "gguf.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>

#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/math_util.hpp"
#include "openvino/util/mmap_object.hpp"
#include "weights.hpp"

namespace ov {
namespace frontend {
namespace gguf {

namespace {

constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF" little-endian
constexpr uint64_t GGUF_DEFAULT_ALIGNMENT = 32;

// (items_per_block, bytes_per_block) per GgufTensorType, indexed by the type id.
struct TypeTraits {
    uint32_t items_per_block;
    uint32_t bytes_per_block;
};

TypeTraits type_traits(uint32_t type) {
    switch (type) {
    case GGUF_TYPE_F32:
        return {1, 4};
    case GGUF_TYPE_F16:
        return {1, 2};
    case GGUF_TYPE_Q4_0:
        return {32, 18};
    case GGUF_TYPE_Q4_1:
        return {32, 20};
    case GGUF_TYPE_Q5_0:
        return {32, 22};
    case GGUF_TYPE_Q5_1:
        return {32, 24};
    case GGUF_TYPE_Q8_0:
        return {32, 34};
    case GGUF_TYPE_Q8_1:
        return {32, 40};
    case GGUF_TYPE_Q2_K:
        return {256, 84};
    case GGUF_TYPE_Q2_0:
        return {64, 18};  // f16 scale + 64x 2-bit codes (16 bytes)
    case GGUF_TYPE_Q3_K:
        return {256, 110};
    case GGUF_TYPE_Q4_K:
        return {256, 144};
    case GGUF_TYPE_Q5_K:
        return {256, 176};
    case GGUF_TYPE_Q6_K:
        return {256, 210};
    case GGUF_TYPE_Q8_K:
        return {256, 292};
    case GGUF_TYPE_I8:
        return {1, 1};
    case GGUF_TYPE_I16:
        return {1, 2};
    case GGUF_TYPE_I32:
        return {1, 4};
    case GGUF_TYPE_I64:
        return {1, 8};
    case GGUF_TYPE_F64:
        return {1, 8};
    case GGUF_TYPE_BF16:
        return {1, 2};
    case GGUF_TYPE_MXFP4:
        return {32, 17};  // 1-byte E8M0 scale + 16 bytes (32x 4-bit)
    default:
        return {0, 0};
    }
}

std::optional<ov::element::Type> gguf_type_to_dtype(uint32_t gguf_type) {
    switch (gguf_type) {
    case GGUF_TYPE_F64:
        return ov::element::f64;
    case GGUF_TYPE_F32:
        return ov::element::f32;
    case GGUF_TYPE_F16:
        return ov::element::f16;
    case GGUF_TYPE_BF16:
        return ov::element::bf16;
    case GGUF_TYPE_I8:
        return ov::element::i8;
    case GGUF_TYPE_I16:
        return ov::element::i16;
    case GGUF_TYPE_I32:
        return ov::element::i32;
    case GGUF_TYPE_I64:
        return ov::element::i64;
    default:
        return std::nullopt;
    }
}

// Sequential little-endian reader over the mmaped file. Bounds-checked on every read.
class Cursor {
public:
    Cursor(const uint8_t* data, uint64_t size) : m_data(data), m_size(size) {}

    uint64_t offset() const {
        return m_off;
    }
    const uint8_t* ptr() const {
        return m_data + m_off;
    }

    // Bytes left before the end of the mapped file. m_off never exceeds m_size (every mutator
    // below asserts that before advancing it), so this cannot underflow.
    uint64_t remaining() const {
        return m_size - m_off;
    }

    template <typename T>
    T read() {
        OPENVINO_ASSERT(sizeof(T) <= remaining(), "[load_gguf] unexpected end of file");
        T v;
        std::memcpy(&v, m_data + m_off, sizeof(T));
        m_off += sizeof(T);
        return v;
    }

    std::string read_string() {
        uint64_t len = read<uint64_t>();
        // Checked against `remaining()` (a subtraction), not `m_off + len` (an addition): len is
        // an attacker-controlled 64-bit value from the file and could otherwise overflow the
        // addition, wrapping it back under m_size and defeating the bounds check.
        OPENVINO_ASSERT(len <= remaining(), "[load_gguf] string runs past end of file");
        std::string s(reinterpret_cast<const char*>(m_data + m_off), static_cast<size_t>(len));
        m_off += len;
        return s;
    }

    void skip(uint64_t n) {
        OPENVINO_ASSERT(n <= remaining(), "[load_gguf] skip past end of file");
        m_off += n;
    }

private:
    const uint8_t* m_data;
    uint64_t m_size;
    uint64_t m_off = 0;
};

size_t value_type_size(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_TYPE_UINT8:
    case GGUF_VALUE_TYPE_INT8:
    case GGUF_VALUE_TYPE_BOOL:
        return 1;
    case GGUF_VALUE_TYPE_UINT16:
    case GGUF_VALUE_TYPE_INT16:
        return 2;
    case GGUF_VALUE_TYPE_UINT32:
    case GGUF_VALUE_TYPE_INT32:
    case GGUF_VALUE_TYPE_FLOAT32:
        return 4;
    case GGUF_VALUE_TYPE_UINT64:
    case GGUF_VALUE_TYPE_INT64:
    case GGUF_VALUE_TYPE_FLOAT64:
        return 8;
    default:
        return 0;  // string / array handled separately
    }
}

ov::element::Type value_type_to_dtype(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_TYPE_UINT8:
        return ov::element::u8;
    case GGUF_VALUE_TYPE_INT8:
        return ov::element::i8;
    case GGUF_VALUE_TYPE_UINT16:
        return ov::element::u16;
    case GGUF_VALUE_TYPE_INT16:
        return ov::element::i16;
    case GGUF_VALUE_TYPE_UINT32:
        return ov::element::u32;
    case GGUF_VALUE_TYPE_INT32:
        return ov::element::i32;
    case GGUF_VALUE_TYPE_FLOAT32:
        return ov::element::f32;
    case GGUF_VALUE_TYPE_BOOL:
        return ov::element::boolean;
    case GGUF_VALUE_TYPE_UINT64:
        return ov::element::u64;
    case GGUF_VALUE_TYPE_INT64:
        return ov::element::i64;
    case GGUF_VALUE_TYPE_FLOAT64:
        return ov::element::f64;
    default:
        OPENVINO_THROW("[load_gguf] unexpected scalar metadata type ", type);
    }
}

// Read a single metadata value of `type` at the cursor into `out`.
void read_metadata_value(Cursor& cur, uint32_t type, GGUFMetaData& out) {
    if (type == GGUF_VALUE_TYPE_STRING) {
        out = cur.read_string();
        return;
    }
    if (type == GGUF_VALUE_TYPE_ARRAY) {
        uint32_t elem_type = cur.read<uint32_t>();
        uint64_t len = cur.read<uint64_t>();
        OPENVINO_ASSERT(elem_type != GGUF_VALUE_TYPE_ARRAY, "[load_gguf] nested arrays are not supported.");
        if (elem_type == GGUF_VALUE_TYPE_STRING) {
            // Each string costs at least 8 bytes (its own length prefix), so this bounds the
            // vector allocation below the same way the fixed-size branch bounds its Tensor.
            OPENVINO_ASSERT(len <= cur.remaining() / sizeof(uint64_t),
                            "[load_gguf] string array length ",
                            len,
                            " overflows the remaining file size");
            std::vector<std::string> strs(len);
            for (auto& s : strs) {
                s = cur.read_string();
            }
            out = std::move(strs);
            return;
        }
        auto dtype = value_type_to_dtype(elem_type);
        const size_t elem_size = value_type_size(elem_type);
        // `len` and `elem_size` are both attacker-controlled (elem_size indirectly, via
        // elem_type) and size a Tensor allocation and a memcpy below; validate `len` against the
        // cursor's remaining bytes -- via a division, not `len * elem_size`, which could itself
        // overflow and wrap under the real remaining size -- before allocating or copying
        // anything.
        OPENVINO_ASSERT(len <= cur.remaining() / elem_size,
                        "[load_gguf] array length ",
                        len,
                        " overflows the remaining file size");
        const size_t nbytes = static_cast<size_t>(len) * elem_size;
        ov::Tensor t(dtype, ov::Shape{static_cast<size_t>(len)});
        std::memcpy(t.data(), cur.ptr(), nbytes);
        cur.skip(nbytes);
        out = std::move(t);
        return;
    }
    // Scalar: store as a shape-{} ov::Tensor of the right element type.
    auto dtype = value_type_to_dtype(type);
    ov::Tensor t(dtype, ov::Shape(0));
    const size_t nbytes = value_type_size(type);
    OPENVINO_ASSERT(nbytes <= cur.remaining(), "[load_gguf] scalar metadata value runs past end of file");
    std::memcpy(t.data(), cur.ptr(), nbytes);
    cur.skip(nbytes);
    out = std::move(t);
}

// Zero-copy view into the mmap for a non-quantized tensor. Uses the dev-API
// Tensor(view, so) constructor so that the mmap shared_ptr is stored in _so and keeps
// the mapping alive for the full lifetime of the returned tensor (and any Constant that
// wraps it via Constant(const Tensor&) -> SharedBuffer<Tensor>).
ov::Tensor extract_tensor_data(const GgufTensor& tensor, const std::shared_ptr<ov::MappedMemory>& mmap) {
    auto dtype = gguf_type_to_dtype(tensor.type);
    OPENVINO_ASSERT(dtype.has_value(),
                    "[load_gguf] tensor '",
                    std::string(tensor.name, tensor.namelen),
                    "' has unsupported non-quantized type ",
                    tensor.type);
    auto shape = get_shape(tensor);
    ov::Tensor view(dtype.value(), shape, const_cast<void*>(static_cast<const void*>(tensor.weights_data)));
    // Attach the mmap shared_ptr as _so so the mapping stays alive for the tensor's lifetime.
    return ov::Tensor(view, mmap);
}

// Fetch a metadata value as an ov::Tensor of the expected element type, failing with the KEY NAME
// on any failure mode: the key is absent, it is present but not a scalar tensor, or its element
// type doesn't match (so a caller can't read past a smaller-than-expected stored type, e.g. a
// 1-byte BOOL where a 4-byte scalar is expected).
static const ov::Tensor& metadata_scalar_tensor(const std::unordered_map<std::string, GGUFMetaData>& metadata,
                                                const std::string& key,
                                                const ov::element::Type& expected_type) {
    auto it = metadata.find(key);
    OPENVINO_ASSERT(it != metadata.end(), "[GGUF] required metadata key is missing: '", key, "'");
    const auto* tensor = std::get_if<ov::Tensor>(&it->second);
    OPENVINO_ASSERT(tensor && tensor->data() && tensor->get_element_type() == expected_type,
                    "[GGUF] metadata key '",
                    key,
                    "' is not a ",
                    expected_type,
                    " scalar as expected");
    return *tensor;
}

float metadata_to_float(const std::unordered_map<std::string, GGUFMetaData>& metadata, const std::string& key) {
    const auto& tensor = metadata_scalar_tensor(metadata, key, ov::element::f32);
    return *(tensor.data<ov::element_type_traits<ov::element::f32>::value_type>());
}

int metadata_to_int(const std::unordered_map<std::string, GGUFMetaData>& metadata, const std::string& key) {
    // GGUF stores these counts as u32; reinterpret as i32 (values fit comfortably).
    const auto& tensor = metadata_scalar_tensor(metadata, key, ov::element::u32);
    return static_cast<int>(*(tensor.data<ov::element_type_traits<ov::element::u32>::value_type>()));
}

// `metadata_to_int`/`metadata_to_float`, but returning `default_value` when `key` is absent
// instead of asserting. Most per-architecture scalars below are optional this way: llama.cpp's
// hparam loading has the same "missing key -> compile-time default" convention per key.
int metadata_to_int_or(const std::unordered_map<std::string, GGUFMetaData>& metadata,
                       const std::string& key,
                       int default_value) {
    return metadata.count(key) ? metadata_to_int(metadata, key) : default_value;
}

float metadata_to_float_or(const std::unordered_map<std::string, GGUFMetaData>& metadata,
                           const std::string& key,
                           float default_value) {
    return metadata.count(key) ? metadata_to_float(metadata, key) : default_value;
}

// GGUF stores true/false flags (e.g. "<arch>.expert_weights_norm") as a 1-byte
// element::boolean scalar tensor, not the u32 metadata_to_int reinterprets; reading it through
// metadata_to_int would read 4 bytes out of a 1-byte allocation. Read the boolean storage
// directly instead.
bool metadata_to_bool_or(const std::unordered_map<std::string, GGUFMetaData>& metadata,
                         const std::string& key,
                         bool default_value) {
    if (!metadata.count(key)) {
        return default_value;
    }
    const auto& tensor = metadata_scalar_tensor(metadata, key, ov::element::boolean);
    return *(tensor.data<ov::element_type_traits<ov::element::boolean>::value_type>()) != 0;
}

}  // namespace

ov::Shape get_shape(const GgufTensor& tensor) {
    ov::Shape shape;
    // GGUF stores dimensions fastest-varying first; the logical (GGML) order is reversed.
    for (int i = static_cast<int>(tensor.ndim) - 1; i >= 0; i--) {
        shape.push_back(tensor.dim[i]);
    }
    return shape;
}

GGUFLoad get_gguf_data(const std::string& file) {
    std::unordered_map<std::string, GGUFMetaData> metadata;
    std::unordered_map<std::string, ov::Tensor> arrays;
    std::unordered_map<std::string, GgufTensorType> qtype;

    auto mapped = ov::load_mmap_object(file);
    OPENVINO_ASSERT(mapped && mapped->data(), "[load_gguf] failed to mmap '", file, "'");
    const auto* base = reinterpret_cast<const uint8_t*>(mapped->data());
    const uint64_t fsize = mapped->size();

    Cursor cur(base, fsize);

    // ---- Header ----
    uint32_t magic = cur.read<uint32_t>();
    OPENVINO_ASSERT(magic == GGUF_MAGIC, "[load_gguf] '", file, "' is not a GGUF file (bad magic)");
    uint32_t version = cur.read<uint32_t>();
    OPENVINO_ASSERT(version == 2 || version == 3, "[load_gguf] unsupported GGUF version ", version);
    uint64_t tensor_count = cur.read<uint64_t>();
    uint64_t kv_count = cur.read<uint64_t>();

    // ---- Metadata kv pairs ----
    for (uint64_t i = 0; i < kv_count; i++) {
        std::string key = cur.read_string();
        uint32_t vtype = cur.read<uint32_t>();
        auto& slot = metadata.insert({key, GGUFMetaData{}}).first->second;
        read_metadata_value(cur, vtype, slot);
    }

    uint64_t alignment = GGUF_DEFAULT_ALIGNMENT;
    if (auto it = metadata.find("general.alignment"); it != metadata.end()) {
        if (auto* t = std::get_if<ov::Tensor>(&it->second)) {
            alignment = *(t->data<ov::element_type_traits<ov::element::u32>::value_type>());
        }
    }
    // The GGUF spec requires alignment to be a non-zero power of two; it is also used below as a
    // modulo divisor, so an unvalidated 0 (a perfectly well-formed u32 scalar KV) would be an
    // uncatchable SIGFPE crash in read_model on a crafted or corrupt file.
    OPENVINO_ASSERT(alignment != 0 && (alignment & (alignment - 1)) == 0,
                    "[load_gguf] invalid general.alignment ",
                    alignment);

    // ---- Tensor info section ----
    struct TensorInfo {
        std::string name;
        uint32_t type = 0;
        uint32_t ndim = 0;
        uint64_t dim[4] = {1, 1, 1, 1};
        uint64_t offset = 0;
    };
    // A minimal tensor-info record (0-length name, ndim=0) is 8 (name len) + 4 (ndim) + 4 (type) +
    // 8 (offset) = 24 bytes; bound the file-controlled `tensor_count` by that before sizing the
    // vector, so a tiny malformed file can't request an effectively unbounded allocation.
    OPENVINO_ASSERT(tensor_count <= cur.remaining() / 24,
                    "[load_gguf] tensor count ",
                    tensor_count,
                    " overflows the remaining file size");
    std::vector<TensorInfo> infos(tensor_count);
    for (uint64_t i = 0; i < tensor_count; i++) {
        TensorInfo& ti = infos[i];
        ti.name = cur.read_string();
        ti.ndim = cur.read<uint32_t>();
        OPENVINO_ASSERT(ti.ndim <= 4, "[load_gguf] tensor '", ti.name, "' has unsupported ndim ", ti.ndim);
        for (uint32_t d = 0; d < ti.ndim; d++) {
            ti.dim[d] = cur.read<uint64_t>();
        }
        ti.type = cur.read<uint32_t>();
        ti.offset = cur.read<uint64_t>();
    }

    // Tensor data starts at the next `alignment`-aligned offset after the info section.
    uint64_t data_off = cur.offset();
    if (uint64_t rem = data_off % alignment) {
        data_off += alignment - rem;
    }

    // Helper: for a quantized tensor, compute (weights_bytes, scale_bytes, zp_bytes).
    // Symmetric types (Q4_0, Q8_0, Q5_0, Q6_K): zp_bytes = 0.
    // Asymmetric types (Q4_1, Q4_K): zp u4 packed (same count as scales, half the bytes).
    // Asymmetric Q5_K: zp u8 (one byte per sub-block, same count as scales).
    //
    // Every dim comes straight from the file, so shape products (and the total below) use
    // ov::util::mul_overflow/add_overflow instead of raw `*=`/`+=`: wrapped products could
    // otherwise under-size quant_buf while the fill functions still write the real,
    // attacker-controlled shape into it.
    auto size_prod = [](const ov::Shape& s) -> size_t {
        size_t result = 1;
        for (auto d : s) {
            size_t next = 0;
            OPENVINO_ASSERT(!ov::util::mul_overflow(result, static_cast<size_t>(d), next),
                            "[load_gguf] tensor element count overflows size_t");
            result = next;
        }
        return result;
    };
    auto quant_sizes = [&size_prod](const TensorInfo& ti) -> std::tuple<size_t, size_t, size_t> {
        auto shape = [&]() {
            ov::Shape s;
            for (int i = static_cast<int>(ti.ndim) - 1; i >= 0; --i)
                s.push_back(ti.dim[i]);
            return s;
        }();
        OPENVINO_ASSERT(!shape.empty(), "[load_gguf] tensor '", ti.name, "' is quantized but has rank 0");

        if (ti.type == GGUF_TYPE_Q8_K) {
            // Q8_K: 256 i8 weights + f32 scale + 16 i16 bsums (ignored) per block.
            const size_t nelems = size_prod(shape);
            const size_t n_blocks = nelems / 256;
            return {nelems, n_blocks * sizeof(float), 0};  // w_bytes, s_bytes(f32), no zp
        }

        if (ti.type == GGUF_TYPE_MXFP4) {
            const size_t nelems = size_prod(shape);
            const size_t cols = shape.back();
            OPENVINO_ASSERT(cols != 0, "[load_gguf] tensor '", ti.name, "' has a zero-sized dimension");
            const size_t groups = cols / 32;
            size_t rows = nelems / cols;
            const size_t w_bytes = (nelems + 1) / 2;  // f4e2m1: 4-bit
            const size_t s_bytes = rows * groups;     // f8e8m0: 1 byte/group
            return {w_bytes, s_bytes, 0};
        }

        const size_t w_nelems = size_prod(shape);

        // Q2_K: u2 (4 per byte), 16 sub-blocks of 16 per super-block.
        if (ti.type == GGUF_TYPE_Q2_K) {
            const size_t w_bytes = (w_nelems + 3) / 4;  // u2: 4 values per byte
            auto scale_shape = shape;
            scale_shape.back() /= 16;
            const size_t s_nelems = size_prod(scale_shape);
            return {w_bytes, s_nelems * sizeof(uint16_t), s_nelems};  // zp: u8 per sub-block
        }

        // Q2_0: u2 (4 per byte), one f16 scale per 64 weights, constant zero-point of 1.
        if (ti.type == GGUF_TYPE_Q2_0) {
            const size_t w_bytes = (w_nelems + 3) / 4;  // u2: 4 values per byte
            auto scale_shape = shape;
            scale_shape.back() /= 64;
            const size_t s_nelems = size_prod(scale_shape);
            return {w_bytes, s_nelems * sizeof(uint16_t), s_nelems};  // zp: u8 per block
        }

        // Q3_K: i4 packed (2 per byte), 16 sub-blocks of 16 per super-block.
        if (ti.type == GGUF_TYPE_Q3_K) {
            const size_t w_bytes = (w_nelems + 1) / 2;  // i4: 2 values per byte
            auto scale_shape = shape;
            scale_shape.back() /= 16;
            const size_t s_nelems = size_prod(scale_shape);
            return {w_bytes, s_nelems * sizeof(uint16_t), 0};  // symmetric: no zp
        }

        // Weights: i8 or u8 stored in byte arrays (u32 only for asymmetric 4-bit).
        // 4-bit types: Q4_0(i4), Q4_1(u4 in u32), Q4_K(u4 in u32).
        // 8-bit types: Q8_0(i8), Q5_0(i8), Q5_1(i8), Q5_K(i8), Q6_K(i8).
        const bool is_4bit = (ti.type == GGUF_TYPE_Q4_0 || ti.type == GGUF_TYPE_Q4_1 || ti.type == GGUF_TYPE_Q4_K);
        uint64_t weights_per_byte = is_4bit ? 2 : 1;
        // Q6_K: 16 weights per sub-block (16 sub-blocks × 16 = 256 per super-block).
        uint64_t weights_per_block = (ti.type == GGUF_TYPE_Q6_K) ? 16 : 32;

        size_t w_bytes;
        if (is_4bit) {
            // u32-packed nibbles: floor((n+7)/8)*4 bytes
            w_bytes = ((w_nelems / weights_per_byte) / 4) * sizeof(uint32_t);
        } else {
            // i8 byte per element
            w_bytes = w_nelems;
        }

        auto scale_shape = shape;
        scale_shape.back() /= weights_per_block;
        const size_t s_nelems = size_prod(scale_shape);
        const size_t s_bytes = s_nelems * sizeof(uint16_t);

        // Zero-point bytes:
        //   Symmetric (Q4_0, Q8_0, Q5_0, Q6_K): no zp.
        //   Q4_1, Q4_K: u4 zp — same element count as scales, packed 2/byte.
        //   Q5_K, Q5_1: u8 zp — one byte per sub-block.
        size_t z_bytes = 0;
        if (ti.type == GGUF_TYPE_Q4_1 || ti.type == GGUF_TYPE_Q4_K) {
            z_bytes = (s_nelems + 1) / 2;  // u4 packed
        } else if (ti.type == GGUF_TYPE_Q5_K || ti.type == GGUF_TYPE_Q5_1) {
            z_bytes = s_nelems;  // u8
        }
        return {w_bytes, s_bytes, z_bytes};
    };

    // ---- Pass 1: total bytes needed for all repacked quantized data ----
    size_t total_quant_bytes = 0;
    for (const auto& ti : infos) {
        const bool is_quant = ti.type == GGUF_TYPE_Q4_0 || ti.type == GGUF_TYPE_Q4_1 || ti.type == GGUF_TYPE_Q5_0 ||
                              ti.type == GGUF_TYPE_Q5_1 || ti.type == GGUF_TYPE_Q8_0 || ti.type == GGUF_TYPE_Q2_K ||
                              ti.type == GGUF_TYPE_Q3_K || ti.type == GGUF_TYPE_Q4_K || ti.type == GGUF_TYPE_Q5_K ||
                              ti.type == GGUF_TYPE_Q6_K || ti.type == GGUF_TYPE_Q8_K || ti.type == GGUF_TYPE_MXFP4 ||
                              ti.type == GGUF_TYPE_Q2_0;
        if (!is_quant)
            continue;
        auto [wb, sb, bb] = quant_sizes(ti);
        size_t sum = 0;
        OPENVINO_ASSERT(!ov::util::add_overflow(wb, sb, sum) && !ov::util::add_overflow(sum, bb, sum) &&
                            !ov::util::add_overflow(total_quant_bytes, sum, total_quant_bytes),
                        "[load_gguf] total quantized buffer size overflows size_t");
    }

    // Single allocation for all repacked quantized data (IR-frontend AlignedBuffer pattern).
    auto quant_buf = std::make_shared<ov::AlignedBuffer>(total_quant_bytes > 0 ? total_quant_bytes : 1);

    // ---- Pass 2: materialize tensors, slicing into quant_buf for quantized ones ----
    size_t quant_offset = 0;
    for (const auto& ti : infos) {
        GgufTensor tensor;
        tensor.name = ti.name.data();
        tensor.namelen = ti.name.size();
        tensor.type = ti.type;
        tensor.ndim = ti.ndim;
        uint64_t nelem = 1;
        for (uint32_t d = 0; d < ti.ndim; d++) {
            tensor.dim[d] = ti.dim[d];
            nelem *= ti.dim[d];
        }
        tensor.num_weights = nelem;
        tensor.offset = ti.offset;

        auto tr = type_traits(ti.type);
        OPENVINO_ASSERT(tr.bytes_per_block != 0, "[load_gguf] tensor '", ti.name, "' has unsupported type ", ti.type);
        // ggml requires the innermost (block) dimension to be a whole number of blocks (e.g.
        // ne[0] % 256 == 0 for Q6_K). Without this, a crafted or truncated file's flooring
        // integer division below silently produces a `bsize` that disagrees with the
        // independently-flooring `scale_shape` computed in quant_sizes, letting fill_* write
        // scales past the end of its (too-small) AlignedBuffer slice.
        OPENVINO_ASSERT(ti.dim[0] % tr.items_per_block == 0,
                        "[load_gguf] tensor '",
                        ti.name,
                        "' innermost dimension ",
                        ti.dim[0],
                        " is not a multiple of its block size ",
                        tr.items_per_block);
        tensor.bsize = (nelem / tr.items_per_block) * tr.bytes_per_block;

        // Subtraction-based bounds checks: ti.offset comes straight from the file, so a crafted
        // value must not be allowed to wrap either addition below back into a valid-looking range.
        OPENVINO_ASSERT(data_off <= fsize, "[load_gguf] tensor data section starts past EOF");
        OPENVINO_ASSERT(ti.offset <= fsize - data_off, "[load_gguf] tensor '", ti.name, "' offset runs past EOF");
        uint64_t abs_off = data_off + ti.offset;
        OPENVINO_ASSERT(tensor.bsize <= fsize - abs_off, "[load_gguf] tensor '", ti.name, "' data runs past EOF");
        tensor.weights_data = base + abs_off;

        const std::string& name = ti.name;
        constexpr std::string_view weight_suffix = ".weight";
        const bool has_weight_suffix =
            name.size() >= weight_suffix.size() &&
            name.compare(name.size() - weight_suffix.size(), weight_suffix.size(), weight_suffix) == 0;
        // For tensors ending in ".weight" strip the suffix so make_weight_node keys are
        // "blk.N.attn_q" (not "blk.N.attn_q.weight").  For quantized non-weight tensors
        // (e.g. gpt-oss "blk.N.attn_k.bias") keep the full name as the prefix; the builder
        // looks up scales/qtype as name + ".scales" / name + ".qtype".
        const std::string name_prefix =
            has_weight_suffix ? name.substr(0, name.length() - weight_suffix.length()) : name;
        if (ti.type == GGUF_TYPE_Q4_0) {
            // Symmetric: i4 weights (XORed u4) + f16 scales, no bias tensor.
            auto [wb, sb, bb] = quant_sizes(ti);
            char* buf_ptr = quant_buf->get_ptr<char>();
            auto shape = get_shape(tensor);
            auto scale_shape = shape;
            scale_shape.back() /= 32;

            std::shared_ptr<void> so_buf(quant_buf);
            ov::Tensor w_view(ov::element::i4, shape, static_cast<void*>(buf_ptr + quant_offset));
            ov::Tensor weights(w_view, so_buf);
            quant_offset += wb;
            ov::Tensor s_view(ov::element::f16, scale_shape, static_cast<void*>(buf_ptr + quant_offset));
            ov::Tensor scales(s_view, so_buf);
            quant_offset += sb;

            gguf_fill_sym(tensor, weights, scales);
            mapped->hint_evict(abs_off, tensor.bsize);

            arrays.emplace(name, std::move(weights));
            arrays.emplace(name_prefix + ".scales", std::move(scales));
            qtype.emplace(name_prefix + ".qtype", GGUF_TYPE_Q4_0);
        } else if (ti.type == GGUF_TYPE_Q3_K) {
            // Symmetric: i4 weights (2 per byte) + f16 scales. No zero-point.
            auto [wb, sb, zb] = quant_sizes(ti);
            (void)zb;
            char* buf_ptr = quant_buf->get_ptr<char>();

            auto shape = get_shape(tensor);
            auto scale_shape = shape;
            scale_shape.back() /= 16;  // 16 sub-blocks per super-block

            std::shared_ptr<void> so_buf(quant_buf);
            ov::Tensor w_view(ov::element::i4, shape, static_cast<void*>(buf_ptr + quant_offset));
            ov::Tensor weights(w_view, so_buf);
            quant_offset += wb;
            ov::Tensor s_view(ov::element::f16, scale_shape, static_cast<void*>(buf_ptr + quant_offset));
            ov::Tensor scales(s_view, so_buf);
            quant_offset += sb;

            gguf_fill_sym(tensor, weights, scales);
            mapped->hint_evict(abs_off, tensor.bsize);

            arrays.emplace(name, std::move(weights));
            arrays.emplace(name_prefix + ".scales", std::move(scales));
            qtype.emplace(name_prefix + ".qtype", GGUF_TYPE_Q3_K);
        } else if (ti.type == GGUF_TYPE_Q2_K) {
            // Asymmetric: u2 weights (4 per byte) + f16 scales + u8 zp per sub-block.
            auto [wb, sb, zb] = quant_sizes(ti);
            char* buf_ptr = quant_buf->get_ptr<char>();

            auto shape = get_shape(tensor);
            auto scale_shape = shape;
            scale_shape.back() /= 16;  // 16 sub-blocks per super-block

            std::shared_ptr<void> so_buf(quant_buf);
            ov::Tensor w_view(ov::element::u2, shape, static_cast<void*>(buf_ptr + quant_offset));
            ov::Tensor weights(w_view, so_buf);
            quant_offset += wb;
            ov::Tensor s_view(ov::element::f16, scale_shape, static_cast<void*>(buf_ptr + quant_offset));
            ov::Tensor scales(s_view, so_buf);
            quant_offset += sb;
            // Fractional zp (= min/scale) in the scale's element type -- see the Q4_K/Q5_K branch.
            ov::Tensor zp(scales.get_element_type(), scale_shape);
            quant_offset += zb;

            gguf_fill_asym(tensor, weights, scales, zp);
            mapped->hint_evict(abs_off, tensor.bsize);

            arrays.emplace(name, std::move(weights));
            arrays.emplace(name_prefix + ".scales", std::move(scales));
            arrays.emplace(name_prefix + ".zp", std::move(zp));
            qtype.emplace(name_prefix + ".qtype", GGUF_TYPE_Q2_K);
        } else if (ti.type == GGUF_TYPE_Q2_0) {
            // Ternary: u2 weights (4 per byte) + one f16 scale per 64 + integer zp of exactly 1.
            // ggml packs Q2_0 codes LSB-first, 4 per byte -- the same order OpenVINO's u2 Constant
            // expects -- so the code bytes are copied verbatim, no repacking.
            auto [wb, sb, zb] = quant_sizes(ti);
            char* buf_ptr = quant_buf->get_ptr<char>();

            auto shape = get_shape(tensor);
            auto scale_shape = shape;
            scale_shape.back() /= 64;  // one scale per 64-weight block

            std::shared_ptr<void> so_buf(quant_buf);
            ov::Tensor w_view(ov::element::u2, shape, static_cast<void*>(buf_ptr + quant_offset));
            ov::Tensor weights(w_view, so_buf);
            quant_offset += wb;
            ov::Tensor s_view(ov::element::f16, scale_shape, static_cast<void*>(buf_ptr + quant_offset));
            ov::Tensor scales(s_view, so_buf);
            quant_offset += sb;
            // Integer zero-point: (code - 1) * scale, so zp is the constant 1 for every block.
            ov::Tensor zp(ov::element::u8, scale_shape);
            quant_offset += zb;

            gguf_fill_q2_0(tensor, weights, scales, zp);
            mapped->hint_evict(abs_off, tensor.bsize);

            arrays.emplace(name, std::move(weights));
            arrays.emplace(name_prefix + ".scales", std::move(scales));
            arrays.emplace(name_prefix + ".zp", std::move(zp));
            qtype.emplace(name_prefix + ".qtype", GGUF_TYPE_Q2_0);
        } else if (ti.type == GGUF_TYPE_Q8_K) {
            // Q8_K: 256 weights/block, f32 scale (NOT f16), 16 i16 bsums (ignored).
            // block_q8_K: [f32 d][i8 qs[256]][i16 bsums[16]] = 4+256+32 = 292 bytes.
            auto [wb, sb, zb] = quant_sizes(ti);
            (void)zb;
            char* buf_ptr = quant_buf->get_ptr<char>();

            auto shape = get_shape(tensor);
            auto scale_shape = shape;
            scale_shape.back() /= 256;  // one f32 scale per 256-weight block

            std::shared_ptr<void> so_buf(quant_buf);
            ov::Tensor w_view(ov::element::i8, shape, static_cast<void*>(buf_ptr + quant_offset));
            ov::Tensor weights_t(w_view, so_buf);
            quant_offset += wb;
            ov::Tensor s_view(ov::element::f32, scale_shape, static_cast<void*>(buf_ptr + quant_offset));
            ov::Tensor scales_t(s_view, so_buf);
            quant_offset += sb;

            gguf_fill_sym(tensor, weights_t, scales_t);
            mapped->hint_evict(abs_off, tensor.bsize);

            arrays.emplace(name, std::move(weights_t));
            arrays.emplace(name_prefix + ".scales", std::move(scales_t));
            qtype.emplace(name_prefix + ".qtype", GGUF_TYPE_Q8_K);
        } else if (ti.type == GGUF_TYPE_Q8_0 || ti.type == GGUF_TYPE_Q5_0 || ti.type == GGUF_TYPE_Q6_K) {
            // Symmetric: i8 weights (no u32 packing) + f16 scales. No zero-point.
            auto [wb, sb, zb] = quant_sizes(ti);
            (void)zb;
            char* buf_ptr = quant_buf->get_ptr<char>();

            auto shape = get_shape(tensor);
            const uint64_t weights_per_block = (ti.type == GGUF_TYPE_Q6_K) ? 16 : 32;
            auto scale_shape = shape;
            scale_shape.back() /= weights_per_block;

            std::shared_ptr<void> so_buf(quant_buf);
            ov::Tensor w_view(ov::element::i8, shape, static_cast<void*>(buf_ptr + quant_offset));
            ov::Tensor weights(w_view, so_buf);
            quant_offset += wb;
            ov::Tensor s_view(ov::element::f16, scale_shape, static_cast<void*>(buf_ptr + quant_offset));
            ov::Tensor scales(s_view, so_buf);
            quant_offset += sb;

            gguf_fill_sym(tensor, weights, scales);
            mapped->hint_evict(abs_off, tensor.bsize);

            arrays.emplace(name, std::move(weights));
            arrays.emplace(name_prefix + ".scales", std::move(scales));
            qtype.emplace(name_prefix + ".qtype", static_cast<GgufTensorType>(ti.type));
        } else if (ti.type == GGUF_TYPE_Q4_1 || ti.type == GGUF_TYPE_Q4_K || ti.type == GGUF_TYPE_Q5_K ||
                   ti.type == GGUF_TYPE_Q5_1) {
            // Asymmetric: weights + f16 scales + integer zp.
            // 4-bit (Q4_1, Q4_K): u32-packed u4 weights, u4 zp.
            // 8-bit (Q5_K, Q5_1): i8 weights, u8 zp.
            auto [wb, sb, zb] = quant_sizes(ti);
            char* buf_ptr = quant_buf->get_ptr<char>();

            auto shape = get_shape(tensor);
            const bool is_4bit = (ti.type == GGUF_TYPE_Q4_1 || ti.type == GGUF_TYPE_Q4_K);
            const uint64_t weights_per_block = 32;

            auto weights_shape = shape;
            if (is_4bit)
                weights_shape.back() /= 8;  // u32 packs 8 u4
            auto scale_shape = shape;
            scale_shape.back() /= weights_per_block;

            std::shared_ptr<void> so_buf(quant_buf);
            ov::element::Type w_elem = is_4bit ? ov::element::u32 : ov::element::i8;
            ov::Tensor w_view(w_elem, is_4bit ? weights_shape : shape, static_cast<void*>(buf_ptr + quant_offset));
            ov::Tensor weights(w_view, so_buf);
            quant_offset += wb;
            ov::Tensor s_view(ov::element::f16, scale_shape, static_cast<void*>(buf_ptr + quant_offset));
            ov::Tensor scales(s_view, so_buf);
            quant_offset += sb;
            // Both ingest paths must agree on the zero-point representation; see
            // gguf_zero_point_type in quant/weights.hpp for why it matters.
            const auto zp_elem = gguf_zero_point_type(name, static_cast<GgufTensorType>(ti.type));
            // Only Q4_K's integer zp actually rounds: Q2_0's zero-point is the exact integer 1.
            if (zp_elem == ov::element::u8 && ti.type == GGUF_TYPE_Q4_K) {
                notify_lossy_weight_approximation(LossyWeightApproximation::INTEGER_ZERO_POINT);
            }
            ov::Tensor zp(zp_elem, scale_shape);
            quant_offset += zb;

            gguf_fill_asym(tensor, weights, scales, zp);
            mapped->hint_evict(abs_off, tensor.bsize);

            arrays.emplace(name, std::move(weights));
            arrays.emplace(name_prefix + ".scales", std::move(scales));
            arrays.emplace(name_prefix + ".zp", std::move(zp));
            qtype.emplace(name_prefix + ".qtype", static_cast<GgufTensorType>(ti.type));
        } else if (ti.type == GGUF_TYPE_MXFP4) {
            // MXFP4: slice weight (f4e2m1) + scale (f8e8m0) out of quant_buf.
            auto [wb, sb, dummy_bb] = quant_sizes(ti);
            (void)dummy_bb;
            char* buf_ptr = quant_buf->get_ptr<char>();

            auto shape = get_shape(tensor);
            const size_t cols = shape.back();
            const size_t groups = cols / 32;
            ov::Shape scale_shape = shape;
            scale_shape.back() = groups;

            std::shared_ptr<void> so_buf(quant_buf);
            ov::Tensor w_view(ov::element::f4e2m1, shape, static_cast<void*>(buf_ptr + quant_offset));
            ov::Tensor weights(w_view, so_buf);
            quant_offset += wb;

            ov::Tensor s_view(ov::element::f8e8m0, scale_shape, static_cast<void*>(buf_ptr + quant_offset));
            ov::Tensor scales(s_view, so_buf);
            quant_offset += sb;

            gguf_fill_mxfp4(tensor, weights, scales);
            mapped->hint_evict(abs_off, tensor.bsize);

            constexpr std::string_view weight_suffix = ".weight";
            const std::string prefix = name.substr(0, name.length() - weight_suffix.length());
            arrays.emplace(name, std::move(weights));
            arrays.emplace(prefix + ".scales", std::move(scales));
            qtype.emplace(prefix + ".qtype", GGUF_TYPE_MXFP4);
        } else {
            ov::Tensor loaded = extract_tensor_data(tensor, mapped);  // zero-copy mmap view
            OPENVINO_ASSERT(arrays.emplace(name, loaded).second, "[load_gguf] duplicate tensor name '", name, "'");
            constexpr std::string_view weight_suffix = ".weight";
            if (name.size() >= weight_suffix.size()) {
                const std::string name_prefix = name.substr(0, name.length() - weight_suffix.length());
                qtype.emplace(name_prefix + ".qtype", static_cast<GgufTensorType>(ti.type));
            }
        }
    }

    return {metadata, arrays, qtype, mapped, quant_buf};
}

std::map<std::string, GGUFMetaData> decoder_config_from_meta(
    const std::unordered_map<std::string, GGUFMetaData>& metadata) {
    std::map<std::string, GGUFMetaData> config;
    // The architecture key drives every other key lookup; fail with a clear message (not a bare
    // std::out_of_range / std::bad_variant_access) if the file carries no / a non-string one.
    auto arch_it = metadata.find("general.architecture");
    OPENVINO_ASSERT(arch_it != metadata.end(),
                    "[GGUF] file has no 'general.architecture' metadata key; not a valid GGUF model file");
    const auto* arch_ptr = std::get_if<std::string>(&arch_it->second);
    OPENVINO_ASSERT(arch_ptr, "[GGUF] 'general.architecture' metadata is not a string");
    const std::string arch = *arch_ptr;
    config["architecture"] = arch;
    config["layer_num"] = metadata_to_int(metadata, arch + ".block_count");
    const int head_count = metadata_to_int(metadata, arch + ".attention.head_count");
    // Validate this independently of key_length: head_count is also used by the builder and,
    // when key_length is absent, is the divisor for the head-size fallback below.
    OPENVINO_ASSERT(head_count > 0, "[GGUF] '", arch, ".attention.head_count' must be positive");
    config["head_num"] = head_count;
    config["head_size"] = metadata.count(arch + ".attention.key_length")
                              ? metadata_to_int(metadata, arch + ".attention.key_length")
                              : (metadata_to_int(metadata, arch + ".embedding_length") / head_count);
    {
        const std::string kv_key = arch + ".attention.head_count_kv";
        if (metadata.count(kv_key)) {
            const auto& kv_val = metadata.at(kv_key);
            if (auto* t = std::get_if<ov::Tensor>(&kv_val)) {
                if (t->get_shape().size() == 1 && t->get_shape()[0] > 1) {
                    // Per-layer array: store as a config tensor for use in the builder.
                    config["head_num_kv_per_layer"] = *t;
                    // Global default = the most common value (max over the array).
                    const auto* data = t->data<uint32_t>();
                    int global_kv = static_cast<int>(*std::max_element(data, data + t->get_size()));
                    config["head_num_kv"] = global_kv;
                } else {
                    config["head_num_kv"] = metadata_to_int(metadata, kv_key);
                }
            }
        } else {
            config["head_num_kv"] = metadata_to_int(metadata, arch + ".attention.head_count");
        }
    }
    config["hidden_size"] = metadata_to_int(metadata, arch + ".embedding_length");
    config["max_position_embeddings"] = metadata_to_int_or(metadata, arch + ".context_length", 2048);
    config["rms_norm_eps"] = metadata_to_float(metadata, arch + ".attention.layer_norm_rms_epsilon");
    config["rope_freq_base"] = metadata_to_float_or(metadata, arch + ".rope.freq_base", 10000.0f);
    // Advisory: the dominant quant type of the file. Purely informational -- every weight carries
    // its own type in the tensor info, which is what the dequant path uses. Optional because
    // llama.cpp's own model writer (llama_model_saver, the source of the per-arch test fixtures)
    // does not emit it; requiring it would reject files llama.cpp itself considers valid.
    config["file_type"] = metadata_to_int_or(metadata, "general.file_type", 0);

    // RoPE YaRN scaling: freq_scale = 1/factor (default 1.0 = no scaling), ext_factor = 1.0
    // for YARN type (0.0 otherwise), n_ctx_orig from rope.scaling.original_context_length.
    // Mirrors llama.cpp: hparams.rope_freq_scale_train = 1/ropescale; ext_factor = (yarn?1:0).
    {
        const float ropescale = metadata_to_float_or(metadata, arch + ".rope.scaling.factor", 0.0f);
        config["rope_freq_scale"] = ropescale == 0.0f ? 1.0f : 1.0f / ropescale;

        const bool is_yarn = metadata.count(arch + ".rope.scaling.type") &&
                             std::get<std::string>(metadata.at(arch + ".rope.scaling.type")) == "yarn";
        config["rope_ext_factor"] = is_yarn ? 1.0f : 0.0f;

        // n_ctx_orig: use rope.scaling.original_context_length when present; fall back to
        // context_length (the training context, which is also n_ctx_train in llama.cpp).
        config["rope_n_ctx_orig"] = metadata_to_int_or(metadata,
                                                       arch + ".rope.scaling.original_context_length",
                                                       std::get<int>(config["max_position_embeddings"]));
    }

    // Number of rope dimensions (n_rot); defaults to head_size. Some archs (e.g. partial-
    // rotary models) set it smaller. A value of 0 means "no RoPE" or full rotation —
    // treat as head_size to avoid division-by-zero and empty-vector crashes downstream.
    {
        const int rope_dims = metadata_to_int_or(metadata, arch + ".rope.dimension_count", 0);
        config["rope_dimension_count"] = (rope_dims > 0) ? rope_dims : std::get<int>(config["head_size"]);
    }
    // Gemma4: SWA layers use a smaller head size (and fewer rope dims) than global layers.
    // key_length_swa / value_length_swa / rope.dimension_count_swa default to key_length.
    config["head_size_swa"] =
        metadata_to_int_or(metadata, arch + ".attention.key_length_swa", std::get<int>(config["head_size"]));
    config["rope_dimension_count_swa"] =
        metadata_to_int_or(metadata, arch + ".rope.dimension_count_swa", std::get<int>(config["rope_dimension_count"]));
    // Gemma4: per-layer embedding dimension (0 = absent / not used).
    config["n_embd_per_layer"] = metadata_to_int_or(metadata, arch + ".embedding_length_per_layer_input", 0);
    // Gemma4: number of layers that have their own KV cache (from the start).
    // shared_kv_layers trailing layers reuse KV from the preceding full-KV layer.
    // 0 = all layers have KV (default for non-Gemma4 architectures).
    config["shared_kv_layers"] = metadata_to_int_or(metadata, arch + ".attention.shared_kv_layers", 0);

    // Mixture-of-experts config (0 when dense).
    config["expert_count"] = metadata_to_int_or(metadata, arch + ".expert_count", 0);
    config["expert_used_count"] = metadata_to_int_or(metadata, arch + ".expert_used_count", 0);
    config["expert_feed_forward_length"] = metadata_to_int_or(metadata, arch + ".expert_feed_forward_length", 0);
    // Hybrid MoE: first N layers are dense, remainder use MoE routing.
    // Mirrors llama.cpp hparams.n_layer_dense_lead (deepseek2-ocr, ernie4_5-moe, glm4moe).
    config["n_layer_dense_lead"] = metadata_to_int_or(metadata, arch + ".leading_dense_block_count", 0);
    // Shared (always-active) experts: run in parallel with routed experts, outputs summed.
    // Mirrors llama.cpp hparams.n_expert_shared (deepseek2-ocr, bailingmoe2, exaone-moe).
    config["expert_shared_count"] = metadata_to_int_or(metadata, arch + ".expert_shared_count", 0);

    // Per-architecture scalars (MiniCPM family). MiniCPM bakes these into hparams with
    // backward-compatible defaults when the GGUF lacks the keys (older exports); newer
    // exports carry the keys and override. Other archs default to 1.0 (no-op).
    const bool is_minicpm = arch.rfind("minicpm", 0) == 0;
    // Gemma/Gemma2/Gemma3/Gemma4 scale embeddings by sqrt(n_embd) before the first layer.
    const bool is_gemma = arch == "gemma" || arch == "gemma2" || arch == "gemma3" || arch == "gemma4";
    const float def_embedding_scale =
        is_minicpm ? 12.0f : (is_gemma ? std::sqrt(static_cast<float>(std::get<int>(config["hidden_size"]))) : 1.0f);
    const float def_residual_scale =
        is_minicpm ? 1.4f / std::sqrt(static_cast<float>(std::get<int>(config["layer_num"]))) : 1.0f;
    const float def_logit_scale = is_minicpm ? 256.0f / static_cast<float>(std::get<int>(config["hidden_size"])) : 1.0f;
    config["embedding_scale"] = metadata_to_float_or(metadata, arch + ".embedding_scale", def_embedding_scale);
    config["residual_scale"] = metadata_to_float_or(metadata, arch + ".residual_scale", def_residual_scale);
    config["logit_scale"] = metadata_to_float_or(metadata, arch + ".logit_scale", def_logit_scale);
    // Hybrid linear-attention (Gated DeltaNet) parameters: qwen35 / qwen3next / kimi-linear.
    // 0 for every non-SSM architecture, which is what the builder tests against.
    auto ssm_key = [&](const std::string& k) {
        return metadata_to_int_or(metadata, arch + "." + k, 0);
    };
    config["ssm_conv_kernel"] = ssm_key("ssm.conv_kernel");
    config["ssm_state_size"] = ssm_key("ssm.state_size");
    config["ssm_group_count"] = ssm_key("ssm.group_count");
    config["ssm_time_step_rank"] = ssm_key("ssm.time_step_rank");
    config["ssm_inner_size"] = ssm_key("ssm.inner_size");
    // Every full_attention_interval-th layer is full attention, the rest are recurrent:
    // llama.cpp qwen35 is_recr(il) = (il < n_layer) && ((il + 1) % interval != 0).
    // llama.cpp defaults the interval to 4 when the GGUF omits the key, so match that rather
    // than rejecting the model (llama.cpp src/models/qwen35.cpp load_arch_hparams).
    config["full_attention_interval"] = metadata_to_int_or(metadata, arch + ".full_attention_interval", 4);
    // NextN / MTP: extra decoder blocks stored past the main stack and NOT executed in a normal
    // forward pass. The builder must stop its layer loop before them.
    config["nextn_predict_layers"] = ssm_key("nextn_predict_layers");
    // Explicit per-layer recurrent flags. llama.cpp consults this FIRST and only falls back to
    // full_attention_interval when it is absent (src/models/qwen35.cpp load_arch_hparams).
    {
        std::vector<int32_t> recr;
        const std::string key = arch + ".attention.recurrent_layers";
        if (metadata.count(key)) {
            const auto& t = std::get<ov::Tensor>(metadata.at(key));
            const auto* p = t.data<uint32_t>();
            for (size_t i = 0; i < t.get_size(); ++i)
                recr.push_back(static_cast<int32_t>(p[i]));
        }
        config["recurrent_layer_flags"] = recr;
    }

    // M-RoPE section widths (qwen35 / qwen3vl): 4 per-axis rotary section sizes.
    {
        std::vector<int32_t> sections;
        const std::string key = arch + ".rope.dimension_sections";
        if (metadata.count(key)) {
            const auto& t = std::get<ov::Tensor>(metadata.at(key));
            const auto* p = t.data<uint32_t>();
            for (size_t i = 0; i < t.get_size() && i < 4; ++i)
                sections.push_back(static_cast<int32_t>(p[i]));
        }
        config["rope_sections"] = sections;
    }

    // Optional explicit attention (softmax) scale; 0 -> use 1/sqrt(head_size).
    // Gemma4 uses scale=1.0 (no pre-attn scaling), per llama.cpp hparams.f_attention_scale=1.0.
    // Gemma3 (like gemma/gemma2) uses 1/sqrt(n_embd_head_k); llama.cpp applies it as a
    // Qcur pre-scale with build_attn(scale=1.0), which is numerically 1/sqrt(head_size) --
    // exactly the default branch here, so gemma3 must NOT force scale=1.0.
    const float def_attention_scale = (arch == "gemma4") ? 1.0f : 0.0f;
    config["attention_scale"] = metadata_to_float_or(metadata, arch + ".attention.scale", def_attention_scale);

    // gpt-oss SWA: separate RoPE frequency base for sliding-window attention layers.
    // Defaults to the global rope_freq_base when the key is absent (non-SWA or legacy models),
    // EXCEPT gemma3: llama.cpp's gemma3 loader keeps the struct default 10000.0 when the key is
    // absent (it does not reset to the global base first, unlike gemma2/gemma4/cohere2/etc.), so
    // gemma3 SWA layers rope at freq_base=10000 while global layers use 1000000. See
    // llama.cpp src/models/gemma3.cpp load_arch_hparams.
    const float def_freq_base_swa = (arch == "gemma3") ? 10000.0f : std::get<float>(config["rope_freq_base"]);
    config["rope_freq_base_swa"] = metadata_to_float_or(metadata, arch + ".rope.freq_base_swa", def_freq_base_swa);

    // has_swa: true when the GGUF carries either a sliding_window_pattern or a
    // sliding_window (a finite window length), indicating SWA is active. The builder uses
    // this to add the self_kq_mask_swa input and route SWA layers to the windowed mask.
    // Architectures that always use sinks (gpt-oss) or per-layer flags (gemma4) are handled
    // separately inside the builder; has_swa catches newly-added archs like smollm3.
    //
    // The same "positive finite" value, when present, is also the window's length in tokens
    // (swa_window_size below): a token at position q may then attend to keys in
    // [q - swa_window_size + 1, q], which AdaptToGenAI needs to build a correctly windowed
    // self_kq_mask_swa instead of reusing the full causal mask.
    uint32_t swa_window_value = 0;
    if (metadata.count(arch + ".attention.sliding_window")) {
        const auto& t = std::get<ov::Tensor>(metadata.at(arch + ".attention.sliding_window"));
        const uint32_t v = *t.data<uint32_t>();
        // A value of 0 or UINT32_MAX typically means "no SWA"; treat only positive finite
        // values as a real window.
        if (v > 0 && v < 0xFFFFFFFFu) {
            swa_window_value = v;
        }
    }
    config["has_swa"] = (metadata.count(arch + ".attention.sliding_window_pattern") || swa_window_value > 0) ? 1 : 0;
    config["swa_window_size"] = static_cast<int>(swa_window_value);

    // gpt-oss SWA: alternation period (default 2: even layers are SWA). Matches llama.cpp's
    // set_swa_pattern(swa_period, dense_first=false): il is SWA if il % period < period - 1.
    // Gemma4: sliding_window_pattern is a boolean array (one entry per layer); stored as
    // an ov::Tensor of element::boolean. Detect by checking the variant type.
    if (metadata.count(arch + ".attention.sliding_window_pattern")) {
        const auto& v = metadata.at(arch + ".attention.sliding_window_pattern");
        if (std::holds_alternative<ov::Tensor>(v)) {
            const auto& t = std::get<ov::Tensor>(v);
            if (t.get_element_type() == ov::element::boolean) {
                // Boolean array: convert to vector<int32_t> (1=SWA, 0=global) for the builder.
                const size_t n = t.get_size();
                std::vector<int32_t> swa_flags(n);
                const auto* bdata = t.data<uint8_t>();
                for (size_t i = 0; i < n; ++i)
                    swa_flags[i] = bdata[i] ? 1 : 0;
                config["swa_layer_flags"] = swa_flags;
                config["swa_layer_pattern"] = 0;  // 0 = use per-layer flags, not period
            } else {
                config["swa_layer_pattern"] = metadata_to_int(metadata, arch + ".attention.sliding_window_pattern");
                config["swa_layer_flags"] = std::vector<int32_t>{};
            }
        } else {
            config["swa_layer_pattern"] = metadata_to_int(metadata, arch + ".attention.sliding_window_pattern");
            config["swa_layer_flags"] = std::vector<int32_t>{};
        }
    } else {
        // No explicit pattern key. gemma3 defaults to period 6 (llama.cpp gemma3 load_arch_hparams
        // passes swa_period=6 to get_key_or_arr); gpt-oss and others default to 2.
        config["swa_layer_pattern"] = (arch == "gemma3") ? 6 : 2;
        config["swa_layer_flags"] = std::vector<int32_t>{};
    }

    // gpt-oss MoE: optional per-expert routing weight scale applied after softmax (0 = 1.0 no-op).
    config["expert_weights_scale"] = metadata_to_float_or(metadata, arch + ".expert_weights_scale", 0.0f);

    // qwen3moe/glm4moe/bailingmoe2 MoE: renormalize the selected top-K gate weights to sum to 1
    // (llama.cpp build_moe_ffn's norm_w / "norm_topk_prob"); absent -> no renormalization.
    config["expert_weights_norm"] = metadata_to_bool_or(metadata, arch + ".expert_weights_norm", false) ? 1 : 0;

    // Gemma2 attention soft-cap: tanh(QK^T * (1/cap)) * cap applied inside the attention.
    // 0.0 means no soft-cap (default for all non-Gemma2 architectures).
    config["attn_logit_softcapping"] = metadata_to_float_or(metadata, arch + ".attn_logit_softcapping", 0.0f);

    // Gemma2/Gemma3 final logit soft-cap applied after lm_head: tanh(x/cap)*cap.
    // 0.0 means no soft-cap.
    config["final_logit_softcapping"] = metadata_to_float_or(metadata, arch + ".final_logit_softcapping", 0.0f);

    return config;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
