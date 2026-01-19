// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/tensor_external_data.hpp"

#include <fstream>
#include <sstream>

#include "exceptions.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/log.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace detail {
TensorExternalData::TensorExternalData(const TensorProto& tensor) {
    for (const auto& entry : tensor.external_data()) {
        if (entry.key() == "location") {
            m_data_location = ov::util::sanitize_path(entry.value());
        } else if (entry.key() == "offset") {
            m_offset = std::stoull(entry.value());
        } else if (entry.key() == "length") {
            m_data_length = std::stoull(entry.value());
        } else if (entry.key() == "checksum") {
            m_sha1_digest = entry.value();
        }
    }
#ifdef ENABLE_OPENVINO_DEBUG
    if (m_sha1_digest.size() > 0) {
        OPENVINO_WARN("SHA1 checksum is not supported");
    }
#endif
}
TensorExternalData::TensorExternalData(const std::string& location, size_t offset, size_t size) {
    m_data_location = location;
    m_offset = offset;
    m_data_length = size;
}

Buffer<ov::MappedMemory> TensorExternalData::load_external_mmap_data(const std::string& model_dir,
                                                                     MappedMemoryHandles cache) const {
    const auto full_path = model_dir.empty() ? ov::util::make_path(m_data_location)
                                             : ov::util::make_path(ov::util::path_join({model_dir, m_data_location}));
    const int64_t file_size = ov::util::file_size(full_path);
    if (file_size <= 0 || m_offset + m_data_length > static_cast<uint64_t>(file_size)) {
        throw error::invalid_external_data{*this};
    }
    auto cached_mapped_memory = cache->find(ov::util::path_to_string(full_path));
    std::shared_ptr<ov::MappedMemory> mapped_memory;
    if (cached_mapped_memory != cache->end()) {
        mapped_memory = cached_mapped_memory->second;
    } else {
        mapped_memory = ov::load_mmap_object(full_path);
        (*cache)[ov::util::path_to_string(full_path)] = mapped_memory;
    }
    if (m_data_length > mapped_memory->size() || mapped_memory->size() == 0) {
        throw error::invalid_external_data{*this};
    }
    return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
        mapped_memory->data() + m_offset,
        m_data_length > 0 ? m_data_length : static_cast<uint64_t>(file_size) - m_offset,
        mapped_memory);
}

Buffer<ov::AlignedBuffer> TensorExternalData::load_external_data(const std::string& model_dir) const {
    const auto full_path = model_dir.empty() ? ov::util::make_path(m_data_location)
                                             : std::filesystem::absolute(std::filesystem::weakly_canonical(
                                                   ov::util::path_join({model_dir, m_data_location})));
    std::ifstream external_data_stream(full_path, std::ios::binary | std::ios::in | std::ios::ate);

    if (external_data_stream.fail()) {
        throw error::invalid_external_data{*this};
    }
    const uint64_t file_size = static_cast<uint64_t>(external_data_stream.tellg());
    if (m_offset + m_data_length > file_size) {
        throw error::invalid_external_data{*this};
    }

    uint64_t read_data_length = m_data_length > 0 ? m_data_length : static_cast<uint64_t>(file_size) - m_offset;

    // default value of m_offset is 0
    external_data_stream.seekg(m_offset, std::ios::beg);

    auto read_data = std::make_shared<ov::AlignedBuffer>(read_data_length);
    external_data_stream.read(read_data->get_ptr<char>(), read_data_length);
    external_data_stream.close();

    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(read_data->get_ptr<char>(),
                                                                                         read_data->size(),
                                                                                         read_data);

    return buffer;
}

Buffer<ov::AlignedBuffer> TensorExternalData::load_external_mem_data() const {
    if (m_data_location != ORT_MEM_ADDR) {
        throw error::invalid_external_data{*this};
    }
    // Empty node will create a constant with zero shape and zero size external data.
    bool is_valid_buffer = m_offset && m_data_length;
    bool is_empty_buffer = (m_data_length == 0);
    if (!(is_valid_buffer || is_empty_buffer)) {
        throw error::invalid_external_data{*this};
    }
    char* addr_ptr = reinterpret_cast<char*>(m_offset);
    auto aligned_memory = std::make_shared<ov::AlignedBuffer>(m_data_length);
    if (m_data_length > 0) {
        std::memcpy(aligned_memory->get_ptr<char>(), addr_ptr, m_data_length);
    }
    return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(aligned_memory->get_ptr<char>(),
                                                                                  aligned_memory->size(),
                                                                                  aligned_memory);
}

std::string TensorExternalData::to_string() const {
    std::stringstream s;
    s << "ExternalDataInfo(";
    s << "data_full_path: " << m_data_location;
    s << ", offset: " << m_offset;
    s << ", data_length: " << m_data_length;
    if (m_sha1_digest.size() > 0) {
        s << ", sha1_digest: " << m_sha1_digest << ")";
    } else {
        s << ")";
    }
    return s.str();
}
}  // namespace detail
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
