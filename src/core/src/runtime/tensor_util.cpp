// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/tensor_util.hpp"

#include <filesystem>
#include <fstream>

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/file_path.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {

namespace {

class StaticBufferAllocator {
    std::shared_ptr<ov::AlignedBuffer> buffer;
    bool allocated = false;  // if buffer was returned as allocated region

public:
    StaticBufferAllocator(std::shared_ptr<ov::AlignedBuffer> _buffer) : buffer(_buffer) {}

    void* allocate(const size_t bytes, const size_t alignment) {
        OPENVINO_ASSERT(alignment == alignof(max_align_t) || alignment == 0);
        OPENVINO_ASSERT(!allocated);
        OPENVINO_ASSERT(bytes == buffer->size());
        allocated = true;
        return buffer->get_ptr();
    }

    void deallocate(void* handle, const size_t bytes, const size_t alignment) {}

    bool is_equal(const StaticBufferAllocator&) const {
        return true;
    }
};
}  // namespace

void save_tensor_data(const ov::Tensor& tensor, const std::filesystem::path& file_name) {
    OPENVINO_ASSERT(tensor.get_element_type() != ov::element::string);
    const char* data = reinterpret_cast<char*>(tensor.data());
    std::ofstream file(file_name, std::ios::binary);
    file.write(data, tensor.get_byte_size());
}
void save_tensor_data(const ov::Tensor& tensor, const std::string& file_name) {
    save_tensor_data(tensor, std::filesystem::path(file_name));
}
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
void save_tensor_data(const Tensor& tensor, const std::wstring& file_name) {
    save_tensor_data(tensor, std::filesystem::path(file_name));
}
#endif

void read_tensor_data(const std::filesystem::path& file_name, Tensor& tensor, size_t offset) {
    OPENVINO_ASSERT(tensor.get_element_type() != ov::element::string);
    std::ifstream fin(file_name, std::ios::binary);
    fin.seekg(offset);
    auto bytes_to_read = tensor.get_byte_size();
    fin.read(static_cast<char*>(tensor.data()), bytes_to_read);
    OPENVINO_ASSERT(static_cast<size_t>(fin.gcount()) == bytes_to_read,
                    "Cannot read ",
                    bytes_to_read,
                    "bytes from ",
                    file_name);
}
void read_tensor_data(const std::string& file_name, Tensor& tensor, size_t offset) {
    read_tensor_data(std::filesystem::path(file_name), tensor, offset);
}
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
void read_tensor_data(const std::wstring& file_name, Tensor& tensor, size_t offset) {
    read_tensor_data(std::filesystem::path(file_name), tensor, offset);
}
#endif

Tensor read_tensor_data(const std::filesystem::path& file_name,
                        const element::Type& element_type,
                        const PartialShape& shape,
                        size_t offset,
                        bool mmap) {
    OPENVINO_ASSERT(element_type != ov::element::string);
    OPENVINO_ASSERT(shape.is_static(), "Cannot read dynamic shape");
    if (mmap) {
        auto mapped_memory = ov::load_mmap_object(file_name);
        auto mmaped = std::make_shared<ov::SharedBuffer<std::shared_ptr<MappedMemory>>>(mapped_memory->data() + offset,
                                                                                        mapped_memory->size() - offset,
                                                                                        mapped_memory);
        return Tensor(element_type, shape.get_shape(), StaticBufferAllocator(mmaped));
    } else {
        ov::Tensor tensor(element_type, shape.get_shape());
        read_tensor_data(file_name, tensor, offset);
        return tensor;
    }
}
Tensor read_tensor_data(const std::string& file_name,
                        const element::Type& element_type,
                        const PartialShape& shape,
                        size_t offset,
                        bool mmap) {
    return read_tensor_data(std::filesystem::path(file_name), element_type, shape, offset, mmap);
}
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
Tensor read_tensor_data(const std::wstring& file_name,
                        const element::Type& element_type,
                        const PartialShape& shape,
                        size_t offset,
                        bool mmap) {
    return read_tensor_data(std::filesystem::path(file_name), element_type, shape, offset, mmap);
}
#endif

namespace {
Tensor read_tensor_data_from_temporary_file(const std::filesystem::path& file_name,
                                            const element::Type& element_type,
                                            const Shape& shape) {
    OPENVINO_ASSERT(element_type != ov::element::string);

    auto mapped_memory = ov::load_mmap_object(file_name);
    auto deleter = [file_name](std::shared_ptr<MappedMemory>& mapped_memory) {
        mapped_memory.reset();
        std::filesystem::remove(file_name);
    };

    using Buffer = ov::RaiiSharedBuffer<std::shared_ptr<MappedMemory>, decltype(deleter)>;
    auto mmaped = std::make_shared<Buffer>(mapped_memory->data(), mapped_memory->size(), mapped_memory, deleter);

    return Tensor(element_type, shape, Allocator(StaticBufferAllocator(mmaped)));
}
}  // namespace

Tensor create_mmaped_tensor(const Tensor& tensor, const std::filesystem::path& file_name) {
    save_tensor_data(tensor, file_name);
    return read_tensor_data_from_temporary_file(file_name, tensor.get_element_type(), tensor.get_shape());
}
Tensor create_mmaped_tensor(const Tensor& tensor, const std::string& file_name) {
    return create_mmaped_tensor(tensor, std::filesystem::path(file_name));
}
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
Tensor create_mmaped_tensor(const Tensor& tensor, const std::wstring& file_name) {
    return create_mmaped_tensor(tensor, std::filesystem::path(file_name));
}
#endif
}  // namespace ov