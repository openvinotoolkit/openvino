// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/tensor.hpp"

#include <numeric>

#include "openvino/core/except.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/core/tensor_util.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/remote_tensor.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {

#define OV_TENSOR_STATEMENT(...)                                      \
    OPENVINO_ASSERT(_impl != nullptr, "Tensor was not initialized."); \
    try {                                                             \
        __VA_ARGS__;                                                  \
    } catch (const std::exception& ex) {                              \
        OPENVINO_THROW(ex.what());                                    \
    } catch (...) {                                                   \
        OPENVINO_ASSERT(false, "Unexpected exception");               \
    }

void Tensor::type_check(const Tensor&) {}

Tensor::~Tensor() {
    _impl = {};
}

Tensor::Tensor(const Tensor& tensor, const std::shared_ptr<void>& so) : _impl{tensor._impl}, _so{tensor._so} {
    OPENVINO_ASSERT(_impl != nullptr, "Tensor was not initialized.");
    if (!_so)
        _so = so;
}

Tensor::Tensor(const std::shared_ptr<ITensor>& impl, const std::shared_ptr<void>& so) : _impl{impl}, _so{so} {
    OPENVINO_ASSERT(_impl != nullptr, "Tensor was not initialized.");
}

Tensor::Tensor(const element::Type& element_type, const Shape& shape, const Allocator& allocator)
    : _impl{make_tensor(element_type, shape, allocator)} {}

Tensor::Tensor(const element::Type& element_type, const Shape& shape, void* host_ptr, const Strides& byte_strides)
    : _impl{make_tensor(element_type, shape, host_ptr, byte_strides)} {}

Tensor::Tensor(const Tensor& owner, const Coordinate& begin, const Coordinate& end)
    : _impl{make_tensor(owner._impl, begin, end)},
      _so{owner._so} {}

Tensor::Tensor(const ov::Output<const ov::Node>& port, const Allocator& allocator)
    : Tensor(port.get_element_type(),
             port.get_partial_shape().is_dynamic() ? ov::Shape{0} : port.get_shape(),
             allocator) {}

Tensor::Tensor(const ov::Output<const ov::Node>& port, void* host_ptr, const Strides& byte_strides)
    : Tensor(port.get_element_type(),
             port.get_partial_shape().is_dynamic() ? ov::Shape{0} : port.get_shape(),
             host_ptr,
             byte_strides) {}

const element::Type& Tensor::get_element_type() const {
    OV_TENSOR_STATEMENT(return _impl->get_element_type());
}

void Tensor::set_shape(const ov::Shape& shape) {
    OV_TENSOR_STATEMENT(_impl->set_shape(shape));
}

const Shape& Tensor::get_shape() const {
    OV_TENSOR_STATEMENT(return _impl->get_shape());
}

void Tensor::copy_to(ov::Tensor dst) const {
    OV_TENSOR_STATEMENT(_impl->copy_to(dst._impl));
}

Strides Tensor::get_strides() const {
    OV_TENSOR_STATEMENT(return _impl->get_strides(););
}

size_t Tensor::get_size() const {
    OV_TENSOR_STATEMENT(return _impl->get_size());
}

size_t Tensor::get_byte_size() const {
    OV_TENSOR_STATEMENT(return _impl->get_byte_size(););
}

void* Tensor::data(const element::Type& element_type) const {
    OV_TENSOR_STATEMENT(return _impl->data(element_type));
}

bool Tensor::operator!() const noexcept {
    return !_impl;
}

Tensor::operator bool() const noexcept {
    return (!!_impl);
}

bool Tensor::is_continuous() const {
    OV_TENSOR_STATEMENT(return _impl->is_continuous());
}

namespace {
ov::Shape calc_static_shape_for_file(const std::filesystem::path& file_name,
                                     const ov::element::Type& element_type,
                                     const ov::PartialShape& partial_shape,
                                     size_t offset) {
    auto file_size = std::filesystem::file_size(file_name);
    if (partial_shape.is_static()) {
        auto static_shape = partial_shape.get_shape();
        OPENVINO_ASSERT((ov::shape_size(static_shape)) * element_type.bitwidth() + offset * 8 == file_size * 8,
                        "Cannot fit file size into requested static PartialShape");
        return static_shape;
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
    OPENVINO_ASSERT(slice_size > 0, "Cannot fit file size into requested PartialShape");

    OPENVINO_ASSERT(dynamic_dimension_numbers.size() == 1,
                    "Only one dynamic dimension in input shape is supported but got: ",
                    dynamic_dimension_numbers.size());
    auto& dynamic_dimension = partial_shape_copy[dynamic_dimension_numbers[0]];

    auto file_size_to_read = file_size - offset;

    OPENVINO_ASSERT((file_size_to_read * 8) % element_type.bitwidth() == 0,
                    "cannot fit ",
                    element_type.get_type_name(),
                    " into ",
                    file_size_to_read,
                    " bytes");
    auto elements_to_read = file_size_to_read * 8 / element_type.bitwidth();

    auto new_dimension = ov::Dimension(elements_to_read) / slice_size;
    OPENVINO_ASSERT(dynamic_dimension.compatible(new_dimension), "Cannot fit file size into requested PartialShape");

    dynamic_dimension = std::move(new_dimension);
    return partial_shape_copy.get_shape();
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
}  // namespace

Tensor read_tensor_data(const std::filesystem::path& file_name,
                        const ov::element::Type& element_type,
                        const ov::PartialShape& partial_shape,
                        size_t offset_in_bytes,
                        bool mmap) {
    OPENVINO_ASSERT(element_type != ov::element::string);
    auto static_shape = calc_static_shape_for_file(file_name, element_type, partial_shape, offset_in_bytes);
    if (mmap) {
        auto mapped_memory = ov::load_mmap_object(file_name);
        auto shared_buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
            mapped_memory->data() + offset_in_bytes,
            mapped_memory->size() - offset_in_bytes,
            mapped_memory);

        auto view_tensor = Tensor(element_type, static_shape, shared_buffer->get_ptr());
        auto impl = get_tensor_impl(view_tensor);
        impl._so = shared_buffer;
        view_tensor = make_tensor(impl);
        return view_tensor;
    } else {
        ov::Tensor tensor(element_type, static_shape);
        read_tensor_data(file_name, tensor, offset_in_bytes);
        return tensor;
    }
}
}  // namespace ov
