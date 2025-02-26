// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/tensor_utils.hpp"

#include <filesystem>
#include <fstream>

#include "openvino/core/type/element_iterator.hpp"
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
    const char* data = reinterpret_cast<const char*>(tensor.data());
    std::ofstream fout(file_name, std::ios::binary);
    fout.write(data, tensor.get_byte_size());
}

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

namespace {
size_t get_element_count(const element::Type& type, const size_t mem_size) {
    if (ov::element::is_split_bit_type(type)) {
        constexpr size_t storage_unit_size = 24;
        size_t integer_number_bytes = mem_size * 8 / storage_unit_size;
        return integer_number_bytes / type.bitwidth();
    } else {
        return mem_size * 8 / type.bitwidth();
    }
}
ov::Shape calc_static_shape_for_file(const std::filesystem::path& file_name,
                                     const element::Type& element_type,
                                     const PartialShape& shape,
                                     size_t offset) {
    auto partial_shape = shape;
    auto rank = partial_shape.rank();
    OPENVINO_ASSERT(rank.is_static(), "Rank cannot be dynamic");
    std::vector<size_t> dynamic_dimension_numbers;
    size_t slice_size = 1;
    for (size_t id = 0; id < partial_shape.size(); ++id) {
        if (partial_shape[id].is_dynamic()) {
            dynamic_dimension_numbers.push_back(id);
        } else {
            slice_size *= partial_shape[id].get_min_length();
        }
    }
    OPENVINO_ASSERT(dynamic_dimension_numbers.size() == 1,
                    "Only dynamic PartialShape with 1 dynamic dimension is supported");
    auto& dynamic_dimension = partial_shape[dynamic_dimension_numbers[0]];

    auto file_size = std::filesystem::file_size(file_name);
    OPENVINO_ASSERT(file_size > offset, "Offset is bigger than size of file to read.");
    auto elements_to_read = get_element_count(element_type, file_size - offset);

    auto new_dimension_size = elements_to_read / slice_size;
    OPENVINO_ASSERT(new_dimension_size * slice_size == elements_to_read,
                    "Cannot fit file size into requested PartialShape");

    OPENVINO_ASSERT(static_cast<int>(new_dimension_size) >= dynamic_dimension.get_min_length() &&
                        (static_cast<int>(new_dimension_size) <= dynamic_dimension.get_max_length() ||
                         dynamic_dimension.get_max_length() == -1),
                    "Cannot fit file size into requested PartialShape");

    dynamic_dimension = Dimension(new_dimension_size);
    return partial_shape.get_shape();
}
}  // namespace

Tensor read_tensor_data(const std::filesystem::path& file_name,
                        const element::Type& element_type,
                        const PartialShape& shape,
                        size_t offset,
                        bool mmap) {
    OPENVINO_ASSERT(element_type != ov::element::string);
    ov::Shape static_shape;
    if (shape.is_dynamic()) {
        static_shape = calc_static_shape_for_file(file_name, element_type, shape, offset);
    } else {
        static_shape = shape.get_shape();
    }

    if (mmap) {
        auto mapped_memory = ov::load_mmap_object(file_name);
        auto mmaped = std::make_shared<ov::SharedBuffer<std::shared_ptr<MappedMemory>>>(mapped_memory->data() + offset,
                                                                                        mapped_memory->size() - offset,
                                                                                        mapped_memory);
        return Tensor(element_type, static_shape, StaticBufferAllocator(mmaped));
    } else {
        ov::Tensor tensor(element_type, static_shape);
        read_tensor_data(file_name, tensor, offset);
        return tensor;
    }
}

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

    using Buffer = ov::UniqueSharedBuffer<std::shared_ptr<MappedMemory>, decltype(deleter)>;
    auto mmaped = std::make_shared<Buffer>(mapped_memory->data(), mapped_memory->size(), mapped_memory, deleter);

    return Tensor(element_type, shape, Allocator(StaticBufferAllocator(mmaped)));
}
}  // namespace

Tensor create_mmaped_tensor(const Tensor& tensor, const std::filesystem::path& file_name) {
    save_tensor_data(tensor, file_name);
    return read_tensor_data_from_temporary_file(file_name, tensor.get_element_type(), tensor.get_shape());
}
}  // namespace ov
