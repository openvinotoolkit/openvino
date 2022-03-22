// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>

#include "blob_factory.hpp"  // IE private header
#include "cpp_interfaces/interface/itensor.hpp"
#include "ie_ngraph_utils.hpp"  // IE private header
#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"
#include "runtime/blob_allocator.hpp"

namespace ov {

#define OV_TENSOR_STATEMENT(...)                                      \
    OPENVINO_ASSERT(_impl != nullptr, "Tensor was not initialized."); \
    try {                                                             \
        __VA_ARGS__;                                                  \
    } catch (const std::exception& ex) {                              \
        throw ov::Exception(ex.what());                               \
    } catch (...) {                                                   \
        OPENVINO_ASSERT(false, "Unexpected exception");               \
    }

void Tensor::type_check(const Tensor&) {}

Tensor::~Tensor() {
    _impl = {};
}

Tensor::Tensor(const ITensor::Ptr& impl, const std::shared_ptr<void>& so) : _impl{impl}, _so{so} {
    OPENVINO_ASSERT(_impl != nullptr, "Tensor was not initialized.");
}

Tensor::Tensor(const std::shared_ptr<ie::Blob>& impl, const std::shared_ptr<void>& so)
    : _impl{blob_to_tensor(impl)},
      _so{so} {
    OPENVINO_ASSERT(_impl != nullptr, "Tensor was not initialized.");
}

Tensor::Tensor(const element::Type element_type, const Shape& shape, const Allocator& allocator)
    : _impl{make_tensor(element_type, shape, allocator)} {}

Tensor::Tensor(const element::Type element_type, const Shape& shape, void* host_ptr, const Strides& byte_strides)
    : _impl{make_tensor(element_type, shape, host_ptr, byte_strides)} {}

Tensor::Tensor(const Tensor& owner, const Coordinate& begin, const Coordinate& end)
    : _so{owner._so},
      _impl{make_tensor(owner._impl, begin, end)} {}

element::Type Tensor::get_element_type() const {
    OV_TENSOR_STATEMENT(return _impl->get_element_type());
}

void Tensor::set_shape(const ov::Shape& shape) {
    OV_TENSOR_STATEMENT(return _impl->set_shape(shape));
}

Shape Tensor::get_shape() const {
    OV_TENSOR_STATEMENT(return _impl->get_shape());
}

Strides Tensor::get_strides() const {
    OV_TENSOR_STATEMENT(return _impl->get_strides());
}

size_t Tensor::get_size() const {
    OV_TENSOR_STATEMENT(return _impl->get_size());
}

size_t Tensor::get_byte_size() const {
    OV_TENSOR_STATEMENT(return _impl->get_byte_size(););
}

void* Tensor::data(const element::Type element_type) const {
    OV_TENSOR_STATEMENT(return _impl->data(element_type));
}

bool Tensor::operator!() const noexcept {
    return !_impl;
}

Tensor::operator bool() const noexcept {
    return (!!_impl);
}

}  // namespace ov
