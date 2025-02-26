// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/tensor_util.hpp"

#include <filesystem>
#include <fstream>

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {

namespace {

struct StaticBufferAllocator {
    std::shared_ptr<ov::AlignedBuffer> buffer;
    bool allocated = false;  // if buffer was returned as allocated region

    StaticBufferAllocator(std::shared_ptr<ov::AlignedBuffer> _buffer) : buffer(_buffer) {}

    void* allocate(const size_t bytes, const size_t alignment) {
        // TODO: Add check for alignment
        OPENVINO_ASSERT(!allocated);
        OPENVINO_ASSERT(bytes == buffer->size());
        allocated = true;
        return buffer->get_ptr();
    }

    void deallocate(void* handle, const size_t bytes, const size_t alignment) {
        OPENVINO_ASSERT(handle == buffer->get_ptr());
        OPENVINO_ASSERT(bytes == buffer->size());
    }

    bool is_equal(const StaticBufferAllocator&) const {
        return true;
    }
};

void save_tensor_data(const ov::Tensor& tensor, const std::string& file_name) {
    OPENVINO_ASSERT(tensor.get_element_type() != ov::element::string);
    const char* data = reinterpret_cast<char*>(tensor.data());
    std::ofstream file(file_name, std::ios::binary | std::ios::out);
    file.write(data, tensor.get_byte_size());
}

Tensor read_tensor_data(const std::string& file_name,
                        const element::Type& element_type,
                        const Shape& shape) {
    OPENVINO_ASSERT(element_type != ov::element::string);

    auto mapped_memory = ov::load_mmap_object(file_name);
    auto deleter = [&file_name] (std::shared_ptr<MappedMemory>& mapped_memory) {
        mapped_memory.reset();
        std::filesystem::remove(file_name);
    };

    using Buffer = ov::RaiiSharedBuffer<std::shared_ptr<MappedMemory>, decltype(deleter)>;
    auto mmaped = std::make_shared<Buffer>(mapped_memory->data(), mapped_memory->size(), mapped_memory, deleter);

    return Tensor(element_type, shape, Allocator(StaticBufferAllocator(mmaped)));
}
}  // namespace

Tensor create_mmaped_tensor(const Tensor& tensor, const std::string& file_name) {
    save_tensor_data(tensor, file_name);
    return read_tensor_data(file_name, tensor.get_element_type(), tensor.get_shape());
}
}  // namespace ov