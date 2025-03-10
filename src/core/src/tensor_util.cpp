// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/tensor_util.hpp"

#include "openvino/core/type/element_iterator.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/file_path.hpp"
#include "openvino/util/mmap_object.hpp"

ov::Tensor ov::util::greater_equal(const ov::Tensor& lhs, const ov::Tensor& rhs) {
    if (!lhs || !rhs)
        return {};
    TensorVector outputs{{element::boolean, {}}};
    if (ov::op::v1::GreaterEqual().evaluate(outputs, {lhs, rhs}))
        return std::move(outputs[0]);
    else
        return {};
}

bool ov::util::reduce_and(const ov::Tensor& t) {
    if (!t)
        return false;

    auto outputs = TensorVector{{element::boolean, Shape{}}};
    auto axes = Tensor(element::i64, Shape{t.get_shape().size()});
    std::iota(axes.data<int64_t>(), axes.data<int64_t>() + t.get_shape().size(), 0);
    if (!ov::op::v1::ReduceLogicalAnd().evaluate(outputs, {t, std::move(axes)}))
        return false;
    return outputs[0].data<char>();
}

namespace ov::util {
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

size_t get_element_count(const ov::element::Type& type, const size_t mem_size) {
    if (ov::element::is_split_bit_type(type)) {
        constexpr size_t storage_unit_size = 24;
        size_t integer_number_bytes = mem_size * 8 / storage_unit_size;
        return integer_number_bytes / type.bitwidth();
    } else {
        return mem_size * 8 / type.bitwidth();
    }
}

ov::Shape calc_static_shape_for_file(const std::filesystem::path& file_name,
                                     const ov::element::Type& element_type,
                                     const ov::PartialShape& partial_shape,
                                     size_t offset) {
    if (partial_shape.is_static()) {
        return partial_shape.get_shape();
    }
    auto partial_shape_copy = partial_shape;
    auto rank = partial_shape_copy.rank();
    OPENVINO_ASSERT(rank.is_static(), "Rank cannot be dynamic");
    std::vector<size_t> dynamic_dimension_numbers;
    size_t slice_size = 1;
    for (size_t id = 0; id < partial_shape_copy.size(); ++id) {
        if (partial_shape_copy[id].is_dynamic()) {
            dynamic_dimension_numbers.push_back(id);
        } else {
            slice_size *= partial_shape_copy[id].get_min_length();
        }
    }
    OPENVINO_ASSERT(dynamic_dimension_numbers.size() == 1,
                    "Only one dynamic dimension in input shape is supported but got: ",
                    dynamic_dimension_numbers.size());
    auto& dynamic_dimension = partial_shape_copy[dynamic_dimension_numbers[0]];

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
    return partial_shape_copy.get_shape();
}
}  // namespace

Tensor read_tensor_data(const std::filesystem::path& file_name,
                        const ov::element::Type& element_type,
                        const ov::PartialShape& partial_shape,
                        size_t offset_in_bytes) {
    OPENVINO_ASSERT(element_type != ov::element::string);
    auto static_shape = calc_static_shape_for_file(file_name, element_type, partial_shape, offset_in_bytes);

    auto mapped_memory = ov::load_mmap_object(file_name);
    auto mmaped =
        std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(mapped_memory->data() + offset_in_bytes,
                                                                              mapped_memory->size() - offset_in_bytes,
                                                                              mapped_memory);
    return ov::Tensor(element_type, static_shape, StaticBufferAllocator(mmaped));
}
}  // namespace ov::util
