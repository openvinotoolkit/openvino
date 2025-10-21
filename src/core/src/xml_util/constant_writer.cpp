// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/xml_util/constant_writer.hpp"

#include "openvino/core/except.hpp"
#include "openvino/reference/convert.hpp"
#include "openvino/runtime/compute_hash.hpp"
#include "openvino/util/common_util.hpp"

namespace ov::util {

std::streamsize OstreamHashWrapperBin::xsputn(const char* s, std::streamsize n) {
    m_res = u64_hash_combine(m_res, *reinterpret_cast<const uint64_t*>(s));
    return n;
}

ConstantWriter::ConstantWriter(std::ostream& bin_data, bool enable_compression)
    : m_binary_output(bin_data),
      m_enable_compression(enable_compression),
      m_write_hash_value(static_cast<bool>(dynamic_cast<OstreamHashWrapperBin*>(bin_data.rdbuf()))),
      m_blob_offset(bin_data.tellp()) {}

ConstantWriter::~ConstantWriter() = default;

ConstantWriter::FilePosition ConstantWriter::write(const char* ptr,
                                                   size_t size,
                                                   size_t& new_size,
                                                   bool compress_to_fp16,
                                                   ov::element::Type src_type,
                                                   bool ptr_is_temporary) {
    // when true, do not rely on ptr after this function call, data
    // is temporary allocated
    const FilePosition write_pos = m_binary_output.get().tellp();
    const auto offset = write_pos - m_blob_offset;
    new_size = size;

    if (!m_enable_compression) {
        if (!compress_to_fp16) {
            m_binary_output.get().write(ptr, size);
        } else {
            OPENVINO_ASSERT(size % src_type.size() == 0);
            auto fp16_buffer = compress_data_to_fp16(ptr, size, src_type, new_size);
            m_binary_output.get().write(fp16_buffer.get(), new_size);
        }
        return offset;
    } else {
        std::unique_ptr<char[]> fp16_buffer = nullptr;
        if (compress_to_fp16) {
            OPENVINO_ASSERT(size % src_type.size() == 0);
            fp16_buffer = compress_data_to_fp16(ptr, size, src_type, new_size);
        }
        const char* ptr_to_write;
        if (fp16_buffer) {
            ptr_to_write = fp16_buffer.get();
        } else {
            ptr_to_write = ptr;
        }

        // This hash is weak (but efficient). For example current hash algorithms gives
        // the same hash for {2, 2} and {0, 128} arrays.
        // But even strong hashing algorithms sometimes give collisions.
        // Therefore we always have to compare values when finding a match in the hash multimap.
        const HashValue hash = ov::runtime::compute_hash(ptr_to_write, new_size);

        auto found = m_hash_to_file_positions.equal_range(hash);
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
        if (m_write_hash_value) {
            m_binary_output.get().write(reinterpret_cast<const char*>(&hash), sizeof(uint64_t));
        } else {
            m_binary_output.get().write(ptr_to_write, new_size);
        }
    }
    return offset;
}

std::unique_ptr<char[]> ConstantWriter::compress_data_to_fp16(const char* ptr,
                                                              size_t size,
                                                              ov::element::Type src_type,
                                                              size_t& compressed_size) {
    auto num_src_elements = size / src_type.size();
    compressed_size = num_src_elements * ov::element::f16.size();
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
