// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/xml_util/weights_provider.hpp"

#include <limits>

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"
#include "openvino/util/parallel_io.hpp"

namespace ov::util {

namespace {

// CVS-193798: unreachable on purpose, mmap must stay disabled here.
size_t get_mmap_region_threshold() {
    return std::numeric_limits<size_t>::max();
}

}  // namespace

std::filesystem::path WeightsProvider::path() const {
    return {};
}

BufferWeightsProvider::BufferWeightsProvider(std::shared_ptr<ov::AlignedBuffer> weights)
    : m_weights(std::move(weights)) {}

std::shared_ptr<ov::AlignedBuffer> BufferWeightsProvider::make_region(size_t offset, size_t size) {
    OPENVINO_ASSERT(m_weights != nullptr, "Empty weights data in bin file or bin file cannot be found!");
    OPENVINO_ASSERT(offset <= m_weights->size() && size <= m_weights->size() - offset,
                    "Incorrect weights in bin file!");

    auto* data = m_weights->get_ptr<char>() + offset;
    return std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(data, size, m_weights);
}

size_t BufferWeightsProvider::size() const {
    OPENVINO_ASSERT(m_weights != nullptr, "Empty weights data in bin file or bin file cannot be found!");
    return m_weights->size();
}

FileWeightsProvider::FileWeightsProvider(std::filesystem::path weights_path) : m_weights_path(std::move(weights_path)) {
    if (!ov::util::file_exists(m_weights_path)) {
        OPENVINO_THROW("Weights file ", m_weights_path, " cannot be opened!");
    }

    std::streamoff weights_size{};
    ov::util::get_file_handle_and_size(m_weights_path, 0, m_weights_handle, weights_size);
    m_weights_size = static_cast<size_t>(weights_size);
}

FileWeightsProvider::~FileWeightsProvider() {
    ov::util::close_file_handle(m_weights_handle);
}

std::shared_ptr<ov::AlignedBuffer> FileWeightsProvider::make_region(size_t offset, size_t size) {
    OPENVINO_ASSERT(offset <= m_weights_size && size <= m_weights_size - offset, "Incorrect weights in bin file!");

    const FileWeightsProvider::WeightsRegionKey key{offset, size};
    if (const auto found = m_loaded_weights_regions.find(key); found != m_loaded_weights_regions.end()) {
        if (auto buffer = found->second.lock()) {
            return buffer;
        }
        m_loaded_weights_regions.erase(found);
    }

    std::shared_ptr<ov::AlignedBuffer> buffer;
    if (size >= get_mmap_region_threshold()) {
        auto mapped_memory = ov::load_mmap_object(m_weights_handle, offset, size);
        buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(mapped_memory->data(),
                                                                                       mapped_memory->size(),
                                                                                       mapped_memory);
    } else {
        auto file_region = std::make_shared<ov::AlignedBuffer>(size);
        if (size > 0) {
            OPENVINO_ASSERT(ov::util::positional_read(m_weights_handle, file_region->get_ptr<char>(), size, offset),
                            "Failed to read weights from ",
                            m_weights_path);
        }
        buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
            file_region->get_ptr<char>(),
            size,
            file_region,
            ov::create_base_descriptor(ov::util::get_id_for_file(m_weights_path, offset, size), offset, file_region));
    }

    m_loaded_weights_regions.emplace(key, buffer);

    return buffer;
}

size_t FileWeightsProvider::size() const {
    return m_weights_size;
}

std::filesystem::path FileWeightsProvider::path() const {
    return m_weights_path;
}
}  // namespace ov::util
