// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/tensor.hpp"

#include <fstream>
#include <numeric>

#include "openvino/core/except.hpp"
#include "openvino/core/memory_util.hpp"
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

Tensor::Tensor(const element::Type& element_type, const Shape& shape, const void* host_ptr, const Strides& byte_strides)
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

Tensor::Tensor(const ov::Output<const ov::Node>& port, const void* host_ptr, const Strides& byte_strides)
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

void* Tensor::data() {
    OV_TENSOR_STATEMENT(return _impl->data_rw());
}

const void* Tensor::data() const {
    OV_TENSOR_STATEMENT(return std::as_const(*_impl).data());
}

void* Tensor::data(const element::Type& element_type) {
    OV_TENSOR_STATEMENT(return _impl->data_rw(element_type));
}

const void* Tensor::data(const element::Type& element_type) const {
    OV_TENSOR_STATEMENT(return std::as_const(*_impl).data(element_type));
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
/**
 * @brief Resolves a static shape from a partial shape and available size in bytes for given element type.
 * @param available_size Size in bytes, must exactly fit the resolved tensor data.
 * @param element_type The element type of the tensor, used to calculate the size of the tensor data.
 * @param partial_shape The partial shape of the tensor. If it has (at most one) dynamic dimension, it will be computed.
 * @return The resolved static shape.
 * @throw ov::Exception if the partial shape has more than one dynamic dimension,
 *        or if the available size is inexact to fit the tensor data of resolved static shape,
 *        or if the dynamic dimension cannot be resolved to fit the available size.
 */
Shape resolve_static_shape(size_t available_size,
                           const element::Type& element_type,
                           const PartialShape& partial_shape) {
    Shape static_shape;
    if (partial_shape.is_static()) {
        static_shape = partial_shape.get_shape();
    } else {
        OPENVINO_ASSERT(partial_shape.rank().is_static(), "Rank cannot be dynamic");

        std::optional<size_t> dynamic_dimension_index;
        typename Dimension::value_type slice_size = 1;
        for (size_t id = 0; id < partial_shape.size(); ++id) {
            if (partial_shape[id].is_dynamic()) {
                OPENVINO_ASSERT(!dynamic_dimension_index,
                                "Only one dynamic dimension in input shape is supported, got ",
                                partial_shape);
                dynamic_dimension_index = id;
            } else {
                slice_size *= partial_shape[id].get_min_length();
            }
        }
        OPENVINO_ASSERT(slice_size > 0, "Cannot fit available bytes into requested PartialShape");

        const auto elements_to_read = util::get_elements_count(element_type, available_size);
        auto new_dimension = Dimension(elements_to_read) / slice_size;
        OPENVINO_ASSERT(partial_shape[*dynamic_dimension_index].compatible(new_dimension),
                        "Cannot fit available bytes into requested PartialShape ",
                        partial_shape);

        auto new_shape = partial_shape;
        new_shape[*dynamic_dimension_index] = std::move(new_dimension);
        static_shape = new_shape.get_shape();
    }

    const auto requested_size = util::get_memory_size_safe(element_type, static_shape);
    OPENVINO_ASSERT(requested_size && *requested_size == available_size,
                    "Requested space exceeds available bounds: available bytes=",
                    available_size,
                    " requested size=",
                    requested_size ? std::to_string(*requested_size) : "uncountable");
    return static_shape;
}

void read_tensor_via_ifstream(const std::filesystem::path& file_name, Tensor& tensor, size_t offset) {
    OPENVINO_ASSERT(tensor.get_element_type() != ov::element::string);
    std::ifstream fin(file_name, std::ios::binary);
    fin.seekg(offset);
    const auto bytes_to_read = static_cast<std::streamsize>(tensor.get_byte_size());
    fin.read(static_cast<char*>(tensor.data()), bytes_to_read);
    OPENVINO_ASSERT(fin.gcount() == bytes_to_read, "Cannot read ", bytes_to_read, " bytes from ", file_name);
}

ov::Tensor wrap_obj_to_viewtensor(const std::shared_ptr<void>& shared_obj,
                                  const void* data_ptr,
                                  const element::Type& element_type,
                                  const Shape& shape) {
    auto view_tensor = Tensor(element_type, shape, data_ptr);
    auto impl = get_tensor_impl(view_tensor);
    impl._so = shared_obj;
    view_tensor = make_tensor(impl);
    return view_tensor;
}

Tensor read_tensor_data_mmap_impl(std::shared_ptr<ov::MappedMemory> mapped_memory,
                                  const ov::element::Type& element_type,
                                  const ov::PartialShape& partial_shape,
                                  size_t offset_in_bytes) {
    const auto static_shape = resolve_static_shape(mapped_memory->size(), element_type, partial_shape);
    const auto shared_buffer =
        std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(mapped_memory->data(),
                                                                              mapped_memory->size(),
                                                                              mapped_memory);
    return wrap_obj_to_viewtensor(shared_buffer, shared_buffer->get_ptr(), element_type, static_shape);
}

size_t get_size_for_mapping(const ov::element::Type& element_type, const ov::PartialShape& partial_shape) {
    if (partial_shape.is_static()) {
        const auto memory_size = ov::util::get_memory_size_safe(element_type, partial_shape.get_shape());
        OPENVINO_ASSERT(memory_size,
                        "Cannot calculate memory size for provided element type ",
                        element_type,
                        " and shape ",
                        partial_shape);
        return *memory_size;
    } else {
        return auto_size;
    }
}
}  // namespace

Tensor read_tensor_data(const std::filesystem::path& file_name,
                        const ov::element::Type& element_type,
                        const ov::PartialShape& partial_shape,
                        size_t offset_in_bytes,
                        bool mmap) {
    OPENVINO_ASSERT(element_type != ov::element::string);
    if (mmap) {
        const auto size = get_size_for_mapping(element_type, partial_shape);
        return read_tensor_data_mmap_impl(load_mmap_object(file_name, offset_in_bytes, size),
                                          element_type,
                                          partial_shape,
                                          offset_in_bytes);
    } else {
        const auto file_size = std::filesystem::file_size(file_name);
        OPENVINO_ASSERT(offset_in_bytes <= file_size,
                        "Requested space exceeds available bounds: offset=",
                        offset_in_bytes,
                        " file size=",
                        file_size);
        const auto available_size = static_cast<size_t>(file_size - offset_in_bytes);
        const auto static_shape = resolve_static_shape(available_size, element_type, partial_shape);
        const auto tensor = std::make_shared<ov::Tensor>(element_type, static_shape);
        read_tensor_via_ifstream(file_name, *tensor.get(), offset_in_bytes);
        return wrap_obj_to_viewtensor(tensor, tensor->data(), element_type, static_shape);
    }
}

Tensor read_tensor_data(ov::FileHandle file_handle,
                        const ov::element::Type& element_type,
                        const ov::PartialShape& partial_shape,
                        size_t offset_in_bytes) {
    OPENVINO_ASSERT(element_type != ov::element::string);
    const auto size = get_size_for_mapping(element_type, partial_shape);
    return read_tensor_data_mmap_impl(load_mmap_object(file_handle, offset_in_bytes, size),
                                      element_type,
                                      partial_shape,
                                      offset_in_bytes);
}
}  // namespace ov
