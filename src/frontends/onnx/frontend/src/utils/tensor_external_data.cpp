// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/tensor_external_data.hpp"

#include <fstream>
#include <sstream>

#include "exceptions.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/log.hpp"
#include "openvino/util/native_stream.hpp"

namespace ov::frontend::onnx::detail {
MappedMemoryHandles make_external_data_cache(bool enable_mmap) {
    return enable_mmap ? std::make_shared<ExternalDataCache>() : nullptr;
}

void set_external_data_reader(MappedMemoryHandles& cache, ExternalDataReader reader) {
    if (!reader) {
        return;
    }
    if (!cache) {
        cache = std::make_shared<ExternalDataCache>();
        cache->enable_mmap = false;
    }
    cache->reader = std::move(reader);
}

TensorExternalData::TensorExternalData(const TensorProto& tensor) {
    for (const auto& entry : tensor.external_data()) {
        if (entry.key() == "location") {
            m_data_location = entry.value();
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

Buffer<ov::MappedMemory> TensorExternalData::load_external_mmap_data(const std::filesystem::path& model_dir,
                                                                     MappedMemoryHandles cache) const {
    std::filesystem::path full_path;
    try {
        full_path = ov::util::sanitize_path(model_dir, ov::util::make_path(m_data_location));
    } catch (const std::runtime_error& e) {
        throw error::invalid_external_data{e.what()};
    }

    const int64_t file_size = ov::util::file_size(full_path);
    if (file_size <= 0 || m_data_length > static_cast<uint64_t>(file_size) ||
        m_offset > static_cast<uint64_t>(file_size) - m_data_length) {
        throw error::invalid_external_data{*this};
    }
    auto cached_mapped_memory = cache->find(full_path);
    std::shared_ptr<ov::MappedMemory> mapped_memory;
    if (cached_mapped_memory != cache->end()) {
        mapped_memory = cached_mapped_memory->second;
    } else {
        mapped_memory = ov::load_mmap_object(full_path);
        (*cache)[full_path] = mapped_memory;
    }
    if (m_data_length > mapped_memory->size() || mapped_memory->size() == 0) {
        throw error::invalid_external_data{*this};
    }
    return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
        mapped_memory->data() + m_offset,
        m_data_length > 0 ? m_data_length : static_cast<uint64_t>(file_size) - m_offset,
        mapped_memory);
}

Buffer<ov::AlignedBuffer> TensorExternalData::load_external_data(const std::filesystem::path& model_dir) const {
    std::filesystem::path full_path;
    try {
        full_path = ov::util::sanitize_path(model_dir, ov::util::make_path(m_data_location));
    } catch (const std::runtime_error& e) {
        throw error::invalid_external_data{e.what()};
    }

    const auto file_size = util::file_size(full_path);
    if (file_size < 0 || m_data_length > static_cast<uint64_t>(file_size) ||
        m_offset > static_cast<uint64_t>(file_size) - m_data_length) {
        throw error::invalid_external_data{*this};
    }

    uint64_t read_data_length = m_data_length > 0 ? m_data_length : static_cast<uint64_t>(file_size) - m_offset;
    auto read_data = std::make_shared<ov::AlignedBuffer>(read_data_length);

    if (read_data_length > 0) {
        util::NativeIfstream external_data_stream(full_path);
        OPENVINO_ASSERT(!external_data_stream.fail(), "Failed to open external data file: ", full_path);

        external_data_stream.seekg(m_offset, std::ios::beg);
        external_data_stream.read(read_data->get_ptr<char>(), read_data_length);
        const auto read_valid =
            external_data_stream && static_cast<size_t>(external_data_stream.gcount()) == read_data_length;
        OPENVINO_ASSERT(read_valid, "Failed to read external data from ", full_path);
    }
    return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(read_data->get_ptr<char>(),
                                                                                  read_data->size(),
                                                                                  read_data);
}

std::shared_ptr<ov::AlignedBuffer> TensorExternalData::load_supplied_data(const std::filesystem::path& model_dir,
                                                                          const MappedMemoryHandles& cache) const {
    if (!cache || !cache->reader) {
        return nullptr;
    }
    std::filesystem::path full_path;
    try {
        full_path = ov::util::sanitize_path(model_dir, ov::util::make_path(m_data_location));
    } catch (const std::runtime_error& e) {
        throw error::invalid_external_data{e.what()};
    }

    auto supplied = cache->supplied.find(full_path);
    if (supplied == cache->supplied.end()) {
        // The reader gets the location relative to the model directory, as declared in the model.
        const auto base_dir = model_dir.empty() ? std::filesystem::current_path() : model_dir;
        const auto location =
            full_path.lexically_relative(std::filesystem::weakly_canonical(base_dir)).generic_string();
        ExternalDataBuffer buffer;
        try {
            buffer = cache->reader(location);
        } catch (const std::exception& e) {
            throw error::invalid_external_data{"The external data reader failed for '" + location + "': " + e.what()};
        }
        supplied = cache->supplied.emplace(full_path, std::move(buffer)).first;
    }

    const auto& buffer = supplied->second;
    if (buffer.data == nullptr) {
        return nullptr;  // Not supplied: read the file
    }
    if (m_offset > buffer.size || m_data_length > buffer.size - m_offset) {
        throw error::invalid_external_data{*this};
    }
    const auto length = m_data_length > 0 ? m_data_length : buffer.size - m_offset;
    auto* data = const_cast<char*>(static_cast<const char*>(buffer.data)) + m_offset;
    return std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(data, static_cast<size_t>(length), buffer.owner);
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
}  // namespace ov::frontend::onnx::detail
