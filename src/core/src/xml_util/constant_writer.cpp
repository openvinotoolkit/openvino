// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/xml_util/constant_writer.hpp"

#include "openvino/core/except.hpp"
#include "openvino/reference/convert.hpp"
#include "openvino/runtime/compute_hash.hpp"
#include "openvino/util/common_util.hpp"

namespace ov::util {

ConstantWriter::ConstantWriter(std::ostream& bin_data, bool enable_compression)
    : m_hash_to_file_positions{},
      m_binary_output(bin_data),
      m_enable_compression(enable_compression),
      m_blob_offset(bin_data.tellp()),
      m_data_hash{} {}

ConstantWriter::~ConstantWriter() = default;

ConstantWriter::FilePosition ConstantWriter::write(const char* ptr,
                                                   size_t size,
                                                   size_t& new_size,
                                                   bool compress_to_fp16,
                                                   ov::element::Type src_type,
                                                   bool ptr_is_temporary) {
    const FilePosition write_pos = m_binary_output.get().tellp();
    const auto offset = write_pos - m_blob_offset;
    new_size = size;

    const auto fp16_data = compress_to_fp16 ? compress_data_to_fp16(ptr, size, src_type, new_size) : nullptr;
    const auto data_ptr = compress_to_fp16 ? fp16_data.get() : ptr;

    if (m_enable_compression) {
        // This hash is weak (but efficient). For example current hash algorithms gives
        // the same hash for {2, 2} and {0, 128} arrays.
        // But even strong hashing algorithms sometimes give collisions.
        // Therefore we always have to compare values when finding a match in the hash multimap.
        const HashValue hash = ov::runtime::compute_hash(data_ptr, new_size);

        const auto found = m_hash_to_file_positions.equal_range(hash);
        // iterate over all matches of the key in the multimap
        for (auto it = found.first; it != found.second; ++it) {
            if (memcmp(ptr, it->second.second, size) == 0) {
                return it->second.first;
            }
        }
        if (!ptr_is_temporary) {
            // Since fp16_compressed data will be disposed at exit point and since we cannot reread it from the
            // ostream, we store pointer to the original uncompressed blob.
            m_hash_to_file_positions.insert({hash, {offset, static_cast<const void*>(ptr)}});
        }
        m_data_hash = util::u64_hash_combine(m_data_hash, hash);
    } else {
        // fast hash (skip data)
        m_data_hash = util::u64_hash_combine(m_data_hash, new_size);
    }
    m_binary_output.get().write(data_ptr, new_size);
    return offset;
}

std::unique_ptr<char[]> ConstantWriter::compress_data_to_fp16(const char* ptr,
                                                              size_t size,
                                                              const element::Type& src_type,
                                                              size_t& compressed_size) {
    auto num_src_elements = size / src_type.size();
    OPENVINO_ASSERT(num_src_elements * src_type.size() == size);
    using T = fundamental_type_for<ov::element::Type_t::f16>;
    compressed_size = num_src_elements * sizeof(T);
    if (src_type == ov::element::f32) {
        auto new_ptr = std::unique_ptr<char[]>(new char[compressed_size]);
        auto dst_data = reinterpret_cast<ov::float16*>(new_ptr.get());
        auto src_data = reinterpret_cast<const float*>(ptr);
        ov::reference::convert_from_f32_to_f16_with_clamp(src_data, dst_data, num_src_elements);
        return new_ptr;
    } else if (src_type == ov::element::f64) {
        auto new_ptr = std::unique_ptr<char[]>(new char[compressed_size]);
        auto dst_data = reinterpret_cast<ov::float16*>(new_ptr.get());
        auto src_data = reinterpret_cast<const double*>(ptr);

        // Reference implementation for fp64 to fp16 conversion
        for (size_t i = 0; i < num_src_elements; ++i) {
            // if abs value is smaller than the smallest positive fp16, but not zero
            if (std::abs(src_data[i]) < ov::float16::from_bits(0x0001) && src_data[i] != 0.0f) {
                dst_data[i] = 0;
            } else if (src_data[i] > std::numeric_limits<ov::float16>::max()) {
                dst_data[i] = std::numeric_limits<ov::float16>::max();
            } else if (src_data[i] < std::numeric_limits<ov::float16>::lowest()) {
                dst_data[i] = std::numeric_limits<ov::float16>::lowest();
            } else {
                dst_data[i] = static_cast<ov::float16>(src_data[i]);
            }
        }
        return new_ptr;
    } else {
        OPENVINO_THROW("[ INTERNAL ERROR ] Not supported source type for weights compression: ", src_type);
    }
}

}  // namespace ov::util
