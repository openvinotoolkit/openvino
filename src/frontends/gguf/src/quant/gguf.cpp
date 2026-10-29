// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Native GGUF container parser. Replaces the third-party gguflib dependency: the file is
// memory-mapped via ov::load_mmap_object and parsed directly per the GGUF v2/v3 format
// (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md). No llama.cpp / ggml
// dependency.

#include "gguf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <unordered_set>

#include "openvino/core/except.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/math_util.hpp"
#include "openvino/util/mmap_object.hpp"
#include "openvino/util/variant_visitor.hpp"
#include "weights.hpp"

namespace ov::frontend::gguf {

namespace {

constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF" little-endian
constexpr uint64_t GGUF_DEFAULT_ALIGNMENT = 32;
constexpr std::string_view WEIGHT_SUFFIX = ".weight";

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
    case GGUF_TYPE_Q1_0:
        return {128, 18};
    case GGUF_TYPE_Q4_1:
        return {32, 20};
    case GGUF_TYPE_Q5_0:
        return {32, 22};
    case GGUF_TYPE_Q5_1:
        return {32, 24};
    case GGUF_TYPE_Q8_0:
        return {32, 34};
    case GGUF_TYPE_Q8_1:
        return {32, 36};
    case GGUF_TYPE_Q2_K:
        return {256, 84};
    case GGUF_TYPE_Q2_0:
        return {64, 18};  // f16 scale + 64x 2-bit codes (16 bytes)
    case GGUF_TYPE_PQ2_0:
        return {128, 34};
    case GGUF_TYPE_PTQ1_0:
        return {128, 28};
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

// Repacked tensor types and shape divisors shared by validation and allocation.
// Non-quantized tensors use mmap views instead.
struct QuantLayout {
    ov::element::Type weight_type;  // element type of the repacked weight blob
    size_t weight_div;              // innermost-dim divisor of the weight shape (8 = u32-packed u4)
    ov::element::Type scale_type;
    size_t group_size;  // weights per scale
    bool asymmetric;    // carries a per-group zero-point
};

std::optional<QuantLayout> quant_layout(uint32_t type, const std::string& name) {
    constexpr bool asymmetric = true;
    constexpr bool symmetric = false;
    switch (type) {
    case GGUF_TYPE_Q4_0:
        return QuantLayout{ov::element::i4, 1, ov::element::f16, 32, symmetric};
    case GGUF_TYPE_Q1_0:
        return QuantLayout{ov::element::u1, 1, ov::element::f16, 128, symmetric};
    case GGUF_TYPE_Q3_K:
        return QuantLayout{ov::element::i4, 1, ov::element::f16, 16, symmetric};
    case GGUF_TYPE_Q5_0:
    case GGUF_TYPE_Q8_0:
    case GGUF_TYPE_Q8_1:
        return QuantLayout{ov::element::i8, 1, ov::element::f16, 32, symmetric};
    case GGUF_TYPE_Q6_K:
        return QuantLayout{ov::element::i8, 1, ov::element::f16, 16, symmetric};
    case GGUF_TYPE_Q8_K:
        // 256 i8 weights + an f32 (not f16) scale + 16 i16 bsums, which the dequant ignores.
        return QuantLayout{ov::element::i8, 1, ov::element::f32, 256, symmetric};
    case GGUF_TYPE_Q2_K:
        return QuantLayout{ov::element::u2, 1, ov::element::f16, 16, asymmetric};
    case GGUF_TYPE_Q2_0:
        return QuantLayout{ov::element::u2, 1, ov::element::f16, 64, asymmetric};
    case GGUF_TYPE_PQ2_0:
    case GGUF_TYPE_PTQ1_0:
        return QuantLayout{ov::element::u2, 1, ov::element::f16, 128, asymmetric};
    case GGUF_TYPE_Q4_1:
    case GGUF_TYPE_Q4_K:
        return QuantLayout{ov::element::u32, 8, ov::element::f16, 32, asymmetric};
    case GGUF_TYPE_Q5_1:
        return QuantLayout{ov::element::i8, 1, ov::element::f16, 32, asymmetric};
    case GGUF_TYPE_Q5_K:
        return native_q5_k_is_symmetric(name) ? QuantLayout{ov::element::i8, 1, ov::element::f16, 32, symmetric}
                                              : QuantLayout{ov::element::u8, 1, ov::element::f16, 32, asymmetric};
    case GGUF_TYPE_MXFP4:
        return QuantLayout{ov::element::f4e2m1, 1, ov::element::f8e8m0, 32, symmetric};
    default:
        return std::nullopt;
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
        const size_t elem_size = dtype.size();
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
    const size_t nbytes = dtype.size();
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

}  // namespace

ov::Shape get_shape(const GgufTensor& tensor) {
    ov::Shape shape;
    // GGUF stores dimensions fastest-varying first; the logical (GGML) order is reversed.
    for (int i = static_cast<int>(tensor.ndim) - 1; i >= 0; i--) {
        shape.push_back(tensor.dim[i]);
    }
    return shape;
}

namespace {

struct TensorInfo {
    std::string name;
    uint32_t type = 0;
    uint32_t ndim = 0;
    uint64_t dim[4] = {1, 1, 1, 1};
    uint64_t offset = 0;
};

struct GgufFile {
    std::unordered_map<std::string, GGUFMetaData> metadata;
    std::vector<TensorInfo> infos;
    std::shared_ptr<ov::MappedMemory> mapped;
    uint64_t alignment;
    uint64_t data_off;
};

GgufFile read_gguf_file(const std::filesystem::path& file) {
    std::unordered_map<std::string, GGUFMetaData> metadata;

    auto mapped = ov::load_mmap_object(file);
    OPENVINO_ASSERT(mapped && mapped->data(), "[load_gguf] failed to mmap '", ov::util::path_to_string(file), "'");
    const auto* base = reinterpret_cast<const uint8_t*>(mapped->data());
    const uint64_t fsize = mapped->size();

    Cursor cur(base, fsize);

    // ---- Header ----
    uint32_t magic = cur.read<uint32_t>();
    OPENVINO_ASSERT(magic == GGUF_MAGIC,
                    "[load_gguf] '",
                    ov::util::path_to_string(file),
                    "' is not a GGUF file (bad magic)");
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
    uint64_t data_off = cur.offset() + ov::util::align_padding_size(alignment, cur.offset());
    OPENVINO_ASSERT(data_off <= fsize, "[load_gguf] tensor data section starts past EOF");
    return {std::move(metadata), std::move(infos), std::move(mapped), alignment, data_off};
}

void detect_legacy_q2_layout(std::vector<GgufFile>& files) {
    std::vector<TensorInfo*> ambiguous_q2;
    bool legacy_g128 = false;
    for (auto& file : files) {
        auto& infos = file.infos;
        const auto alignment = file.alignment;
        const auto data_off = file.data_off;
        const uint64_t fsize = file.mapped->size();
        std::vector<uint64_t> offsets;
        for (const auto& ti : infos)
            offsets.push_back(ti.offset);
        std::sort(offsets.begin(), offsets.end());
        OPENVINO_ASSERT(data_off <= fsize && (offsets.empty() || offsets.back() <= fsize - data_off),
                        "[load_gguf] tensor offset runs past EOF");
        for (auto& ti : infos) {
            if (ti.type != GGUF_TYPE_Q2_0 || ti.dim[0] == 0)
                continue;
            const auto next = std::upper_bound(offsets.begin(), offsets.end(), ti.offset);
            OPENVINO_ASSERT(data_off <= fsize && ti.offset <= fsize - data_off,
                            "[load_gguf] tensor '",
                            ti.name,
                            "' offset runs past EOF");
            const auto span = (next == offsets.end() ? fsize - data_off : *next) - ti.offset;
            const auto size_for = [&](uint32_t type) {
                const auto traits = type_traits(type);
                uint64_t blocks = ti.dim[0] / traits.items_per_block;
                for (uint32_t d = 1; d < ti.ndim; ++d) {
                    OPENVINO_ASSERT(ti.dim[d] == 0 || blocks <= std::numeric_limits<uint64_t>::max() / ti.dim[d],
                                    "[load_gguf] tensor '",
                                    ti.name,
                                    "' byte count overflows");
                    blocks *= ti.dim[d];
                }
                OPENVINO_ASSERT(blocks <= std::numeric_limits<uint64_t>::max() / traits.bytes_per_block,
                                "[load_gguf] tensor '",
                                ti.name,
                                "' byte count overflows");
                const auto bytes = blocks * traits.bytes_per_block;
                return bytes <= span &&
                       (bytes == span || ov::util::align_padding_size(alignment, bytes) == span - bytes);
            };
            const bool g64 = ti.dim[0] % 64 == 0 && size_for(GGUF_TYPE_Q2_0);
            const bool g128 = ti.dim[0] % 128 == 0 && size_for(GGUF_TYPE_PQ2_0);
            OPENVINO_ASSERT(g64 || g128, "[load_gguf] tensor '", ti.name, "' has invalid Q2_0 packing");
            if (g64 && g128) {
                ambiguous_q2.push_back(&ti);
            } else if (g128) {
                ti.type = GGUF_TYPE_PQ2_0;
                legacy_g128 = true;
            }
        }
    }
    if (legacy_g128) {
        for (auto* ti : ambiguous_q2)
            ti->type = GGUF_TYPE_PQ2_0;
    }
}

void load_gguf_tensors(GgufFile& file,
                       std::unordered_map<std::string, ov::Tensor>& arrays,
                       std::unordered_map<std::string, GgufTensorType>& qtype) {
    auto& [metadata, infos, mapped, alignment, data_off] = file;
    const auto* base = reinterpret_cast<const uint8_t*>(mapped->data());
    const uint64_t fsize = mapped->size();

    // Validate file-controlled shapes before allocating repacked tensors.
    auto quant_sizes = [](const TensorInfo& ti, const QuantLayout& ql) -> std::array<size_t, 3> {
        const ov::Shape shape = [&ti]() {
            ov::Shape s;
            for (int i = static_cast<int>(ti.ndim) - 1; i >= 0; --i)
                s.push_back(ti.dim[i]);
            return s;
        }();
        OPENVINO_ASSERT(!shape.empty(), "[load_gguf] tensor '", ti.name, "' is quantized but has rank 0");
        if (ti.type == GGUF_TYPE_MXFP4) {
            OPENVINO_ASSERT(shape.back() != 0, "[load_gguf] tensor '", ti.name, "' has a zero-sized dimension");
        }

        const auto bytes = [&ti](const ov::element::Type& et, const ov::Shape& s) {
            const auto n = ov::util::get_memory_size_safe(et, s);
            OPENVINO_ASSERT(n.has_value(), "[load_gguf] tensor '", ti.name, "' byte count overflows size_t");
            return *n;
        };

        ov::Shape weight_shape = shape;
        weight_shape.back() /= ql.weight_div;
        ov::Shape scale_shape = shape;
        scale_shape.back() /= ql.group_size;
        const size_t z_bytes =
            ql.asymmetric ? bytes(gguf_zero_point_type(ti.name, static_cast<GgufTensorType>(ti.type)), scale_shape) : 0;
        return {bytes(ql.weight_type, weight_shape), bytes(ql.scale_type, scale_shape), z_bytes};
    };

    for (const auto& info : infos) {
        if (const auto layout = quant_layout(info.type, info.name)) {
            quant_sizes(info, *layout);
        }
    }

    for (size_t idx = 0; idx < infos.size(); ++idx) {
        const TensorInfo& ti = infos[idx];
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
        // scales past the end of its allocated tensor.
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
        // For tensors ending in ".weight" strip the suffix so make_weight_node keys are
        // "blk.N.attn_q" (not "blk.N.attn_q.weight").  For quantized non-weight tensors
        // (e.g. gpt-oss "blk.N.attn_k.bias") keep the full name as the prefix; the builder
        // looks up scales/qtype as name + ".scales" / name + ".qtype".
        const std::string name_prefix =
            ov::util::ends_with(name, WEIGHT_SUFFIX) ? name.substr(0, name.length() - WEIGHT_SUFFIX.length()) : name;

        const auto layout = quant_layout(ti.type, ti.name);
        if (!layout) {
            ov::Tensor loaded = extract_tensor_data(tensor, mapped);  // zero-copy mmap view
            const bool inserted = arrays.emplace(name, std::move(loaded)).second;
            OPENVINO_ASSERT(inserted, "[load_gguf] duplicate tensor name '", name, "'");
            qtype.emplace(name_prefix + ".qtype", static_cast<GgufTensorType>(ti.type));
            continue;
        }

        const auto shape = get_shape(tensor);
        ov::Shape weight_shape = shape;
        weight_shape.back() /= layout->weight_div;
        ov::Shape scale_shape = shape;
        scale_shape.back() /= layout->group_size;

        // Independent buffers let replaced weights be freed without retaining the whole model.
        ov::Tensor weights(layout->weight_type, weight_shape);
        ov::Tensor scales(layout->scale_type, scale_shape);
        ov::Tensor zp;
        if (layout->asymmetric) {
            zp = ov::Tensor(gguf_zero_point_type(name, static_cast<GgufTensorType>(ti.type)), scale_shape);
        }

        if (ti.type == GGUF_TYPE_Q5_K && !layout->asymmetric) {
            gguf_fill_q5_k_i8(tensor, weights, scales);
        } else if (ti.type == GGUF_TYPE_MXFP4) {
            gguf_fill_mxfp4(tensor, weights, scales);
        } else if (ti.type == GGUF_TYPE_Q2_0 || ti.type == GGUF_TYPE_PQ2_0 || ti.type == GGUF_TYPE_PTQ1_0) {
            gguf_fill_q2_0(tensor, weights, scales, zp);
        } else if (layout->asymmetric) {
            gguf_fill_asym(tensor, weights, scales, zp);
        } else {
            gguf_fill_sym(tensor, weights, scales);
        }
        mapped->hint_evict(abs_off, tensor.bsize);

        arrays.emplace(name, std::move(weights));
        arrays.emplace(name_prefix + ".scales", std::move(scales));
        if (zp) {
            arrays.emplace(name_prefix + ".zp", std::move(zp));
        }
        qtype.emplace(name_prefix + ".qtype", static_cast<GgufTensorType>(ti.type));
    }
}

uint64_t split_value(const GgufFile& file, const std::string& key, const ov::element::Type& type) {
    const auto& value = metadata_scalar_tensor(file.metadata, key, type);
    OPENVINO_ASSERT(value.get_shape().empty(), "[load_gguf] '", key, "' must be a scalar");
    if (type == ov::element::u16) {
        return *value.data<const uint16_t>();
    }
    const auto count = *value.data<const int32_t>();
    OPENVINO_ASSERT(count >= 0, "[load_gguf] '", key, "' must be nonnegative");
    return static_cast<uint64_t>(count);
}

uint64_t split_count(const GgufFile& file) {
    return file.metadata.count("split.count") ? split_value(file, "split.count", ov::element::u16) : 0;
}

bool same_metadata(const GGUFMetaData& lhs, const GGUFMetaData& rhs) {
    if (lhs.index() != rhs.index()) {
        return false;
    }
    return std::visit(ov::util::VariantVisitor{[&](const ov::Tensor& a) {
                                                   const auto& b = std::get<ov::Tensor>(rhs);
                                                   return a.get_element_type() == b.get_element_type() &&
                                                          a.get_shape() == b.get_shape() &&
                                                          (a.get_byte_size() == 0 ||
                                                           std::memcmp(a.data(), b.data(), a.get_byte_size()) == 0);
                                               },
                                               [&](const auto& a) {
                                                   return a == std::get<std::decay_t<decltype(a)>>(rhs);
                                               }},
                      lhs);
}

std::string split_suffix(uint64_t index, uint64_t count) {
    std::ostringstream suffix;
    suffix << '-' << std::setfill('0') << std::setw(5) << index + 1 << "-of-" << std::setw(5) << count << ".gguf";
    return suffix.str();
}

GGUFLoad load_gguf_files(std::vector<GgufFile> files) {
    const uint64_t count = split_count(files.front());
    if (count != 0) {
        OPENVINO_ASSERT(files.size() == count, "[load_gguf] expected ", count, " GGUF shards, got ", files.size());
        std::vector<bool> seen(count);
        uint64_t tensor_count = 0;
        const auto total = split_value(files.front(), "split.tensors.count", ov::element::i32);
        for (const auto& file : files) {
            OPENVINO_ASSERT(split_count(file) == count, "[load_gguf] inconsistent split.count across GGUF shards");
            const auto index = split_value(file, "split.no", ov::element::u16);
            OPENVINO_ASSERT(index < count, "[load_gguf] split.no is out of range: ", index);
            OPENVINO_ASSERT(!seen[index], "[load_gguf] duplicate GGUF shard index: ", index);
            seen[index] = true;
            OPENVINO_ASSERT(split_value(file, "split.tensors.count", ov::element::i32) == total,
                            "[load_gguf] inconsistent split.tensors.count across GGUF shards");
            tensor_count += file.infos.size();
        }
        OPENVINO_ASSERT(tensor_count == total,
                        "[load_gguf] split.tensors.count declares ",
                        total,
                        " tensors, found ",
                        tensor_count);
        std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) {
            return split_value(a, "split.no", ov::element::u16) < split_value(b, "split.no", ov::element::u16);
        });
    } else {
        OPENVINO_ASSERT(files.size() == 1, "[load_gguf] multiple inputs must belong to one split GGUF model");
    }

    const auto& metadata = files.front().metadata;
    std::unordered_set<std::string> names;
    for (const auto& file : files) {
        for (const auto& [key, value] : file.metadata) {
            if (key.compare(0, 6, "split.") == 0) {
                continue;
            }
            const auto it = metadata.find(key);
            OPENVINO_ASSERT(it != metadata.end() && same_metadata(it->second, value),
                            "[load_gguf] inconsistent metadata across GGUF shards: '",
                            key,
                            "'");
        }
        for (const auto& info : file.infos) {
            OPENVINO_ASSERT(names.insert(info.name).second, "[load_gguf] duplicate tensor name '", info.name, "'");
        }
    }

    std::unordered_map<std::string, ov::Tensor> arrays;
    std::unordered_map<std::string, GgufTensorType> qtype;
    std::vector<std::shared_ptr<ov::MappedMemory>> mappings;
    mappings.reserve(files.size());
    detect_legacy_q2_layout(files);
    for (auto& file : files) {
        load_gguf_tensors(file, arrays, qtype);
        mappings.push_back(std::move(file.mapped));
    }
    return {std::move(files.front().metadata), std::move(arrays), std::move(qtype), std::move(mappings)};
}

}  // namespace

GGUFLoad get_gguf_data(const std::filesystem::path& file) {
    auto initial = read_gguf_file(file);
    const auto count = split_count(initial);
    if (count <= 1) {
        std::vector<GgufFile> files;
        files.push_back(std::move(initial));
        return load_gguf_files(std::move(files));
    }
    const auto index = split_value(initial, "split.no", ov::element::u16);
    OPENVINO_ASSERT(index < count, "[load_gguf] split.no is out of range: ", index);
    const auto filename = file.filename().native();
    const auto suffix = std::filesystem::path(split_suffix(index, count)).native();
    OPENVINO_ASSERT(filename.size() >= suffix.size() &&
                        filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) == 0,
                    "[load_gguf] split GGUF filename does not match its metadata; provide an explicit list of shards");
    const auto prefix = filename.substr(0, filename.size() - suffix.size());
    std::vector<GgufFile> files;
    files.reserve(count);
    for (uint64_t i = 0; i < count; ++i) {
        if (i == index) {
            files.push_back(std::move(initial));
            continue;
        }
        const auto sibling = file.parent_path() / (prefix + std::filesystem::path(split_suffix(i, count)).native());
        OPENVINO_ASSERT(std::filesystem::is_regular_file(sibling),
                        "[load_gguf] missing GGUF shard: '",
                        ov::util::path_to_string(sibling),
                        "'");
        auto next = read_gguf_file(sibling);
        OPENVINO_ASSERT(split_value(next, "split.no", ov::element::u16) == i,
                        "[load_gguf] split.no does not match shard filename: '",
                        ov::util::path_to_string(sibling),
                        "'");
        files.push_back(std::move(next));
    }
    return load_gguf_files(std::move(files));
}

GGUFLoad get_gguf_data(const std::vector<std::filesystem::path>& paths) {
    OPENVINO_ASSERT(!paths.empty(), "[load_gguf] GGUF shard list is empty");
    std::vector<GgufFile> files;
    files.reserve(paths.size());
    for (const auto& path : paths) {
        files.push_back(read_gguf_file(path));
    }
    return load_gguf_files(std::move(files));
}

}  // namespace ov::frontend::gguf
