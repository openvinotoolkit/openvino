// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Self-contained reader for the GGUF binary container format (version 2/3).
//
// Responsibilities:
//   * memory-map (default) or read the file,
//   * parse the header, metadata key/value table and tensor-info table,
//   * expose metadata as ov::Any and build zero-copy ov::op::v0::Constant nodes whose bytes are
//     bound to the underlying mmap region (no dequantization, raw block bytes are preserved).
//
// All model/tensor metadata is treated as untrusted input: every offset/size is bounds-checked
// against the mapped region and the reader fails fast (throws) on truncation or malformed data.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
class MappedMemory;
class AlignedBuffer;
}  // namespace ov

namespace ov {
namespace frontend {
namespace gguf {

struct GGUFTensorInfo {
    std::string name;
    ov::Shape shape;          // OpenVINO logical shape, output-dim-first (GGML dims reversed).
    ov::element::Type type;   // OpenVINO element type (gguf_* block type or f16/f32/bf16/...).
    uint32_t ggml_type = 0;   // raw on-disk GGML type id.
    uint64_t data_offset = 0;  // byte offset of the tensor inside the data section.
    size_t byte_size = 0;      // physical byte size (block-aware).
};

class GGUFReader {
public:
    // Opens and fully parses the container. mmap_enable=false reads the whole file into memory.
    explicit GGUFReader(const std::string& path, bool mmap_enable = true);

    uint32_t version() const {
        return m_version;
    }
    const std::unordered_map<std::string, ov::Any>& metadata() const {
        return m_metadata;
    }
    const std::vector<GGUFTensorInfo>& tensors() const {
        return m_tensors;
    }
    const GGUFTensorInfo* find_tensor(const std::string& name) const;
    bool has_tensor(const std::string& name) const {
        return find_tensor(name) != nullptr;
    }

    // Builds a Constant for the named tensor with its raw (still quantized) bytes bound to the
    // underlying storage. Throws if the tensor does not exist.
    std::shared_ptr<ov::op::v0::Constant> tensor_constant(const std::string& name) const;

    // Returns a pointer to the raw (still quantized) bytes of the named tensor inside the backing
    // storage and sets byte_size to its physical size. The pointer stays valid for the lifetime of
    // the reader. Throws if the tensor does not exist. Used by the (embedding-only) dequantizer.
    const uint8_t* tensor_data(const std::string& name, size_t& byte_size) const;

    // --- metadata access with width-coercion (GGUF stores ints in various widths) ---
    bool has(const std::string& key) const;
    const ov::Any& raw(const std::string& key) const;  // throws if missing
    uint64_t get_u64(const std::string& key) const;
    uint64_t get_u64(const std::string& key, uint64_t default_value) const;
    double get_f64(const std::string& key) const;
    double get_f64(const std::string& key, double default_value) const;
    std::string get_str(const std::string& key) const;
    std::string get_str(const std::string& key, const std::string& default_value) const;

    const std::string& architecture() const {
        return m_architecture;
    }
    const std::string& source_file_hash() const {
        return m_file_hash;
    }

private:
    void parse();
    std::shared_ptr<ov::op::v0::Constant> make_constant(const ov::element::Type& type,
                                                        const ov::Shape& shape,
                                                        const uint8_t* ptr,
                                                        size_t byte_size) const;

    bool m_mmap_enable = true;
    std::shared_ptr<ov::MappedMemory> m_mmap;
    std::shared_ptr<ov::AlignedBuffer> m_buffer;
    const uint8_t* m_base = nullptr;
    size_t m_total_size = 0;
    uint64_t m_data_start = 0;

    uint32_t m_version = 0;
    std::unordered_map<std::string, ov::Any> m_metadata;
    std::vector<GGUFTensorInfo> m_tensors;
    std::unordered_map<std::string, size_t> m_tensor_index;
    std::string m_architecture;
    std::string m_file_hash;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
