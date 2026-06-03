// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/xml_util/weights_provider.hpp"

#include <fstream>

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"

namespace ov::util {

BufferWeightsProvider::BufferWeightsProvider(std::shared_ptr<ov::AlignedBuffer> weights)
    : m_weights(std::move(weights)) {}

std::shared_ptr<ov::AlignedBuffer> BufferWeightsProvider::load_region(size_t offset, size_t size) {
    OPENVINO_ASSERT(m_weights != nullptr, "Empty weights data in bin file or bin file cannot be found!");
    OPENVINO_ASSERT(offset <= m_weights->size() && size <= m_weights->size() - offset,
                    "Incorrect weights in bin file!");

    auto* data = m_weights->get_ptr<char>() + offset;
    return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(data, size, m_weights);
}

size_t BufferWeightsProvider::size() const {
    return m_weights->size();
}

FileWeightsProvider::FileWeightsProvider(std::filesystem::path weights_path)
    : m_weights_path(std::move(weights_path)),
      m_weights_size(ov::util::file_size(m_weights_path)),
      m_weights_source_id(std::filesystem::hash_value(weights_path)),
      m_weights_source_handle(std::make_shared<ov::AlignedBuffer>()) {
    std::ifstream weights_stream(m_weights_path, std::ios::binary);
    OPENVINO_ASSERT(weights_stream.is_open(), "Weights file ", m_weights_path, " cannot be opened!");
}

std::shared_ptr<ov::AlignedBuffer> FileWeightsProvider::load_region(size_t offset, size_t size) {
    OPENVINO_ASSERT(offset <= m_weights_size && size <= m_weights_size - offset, "Incorrect weights in bin file!");

    const FileWeightsProvider::WeightsRegionKey key{offset, size};
    if (const auto found = m_loaded_weights_regions.find(key); found != m_loaded_weights_regions.end()) {
        return found->second;
    }

    std::ifstream weights_stream(m_weights_path, std::ios::binary);
    OPENVINO_ASSERT(weights_stream.is_open(), "Weights file ", m_weights_path, " cannot be opened!");

    weights_stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    OPENVINO_ASSERT(weights_stream.good(), "Failed to seek weights file ", m_weights_path, " to offset ", offset);

    weights_stream.read(m_weights_source_handle->get_ptr<char>(), static_cast<std::streamsize>(size));
    OPENVINO_ASSERT(static_cast<size_t>(weights_stream.gcount()) == size,
                    "Failed to read ",
                    size,
                    " bytes from weights file ",
                    m_weights_path,
                    " at offset ",
                    offset);

    m_loaded_weights_regions.emplace(key, m_weights_source_handle);

    return m_weights_source_handle;
}

size_t FileWeightsProvider::size() const {
    return m_weights_size;
}
}  // namespace ov::util
