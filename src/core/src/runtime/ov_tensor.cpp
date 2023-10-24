// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>

#include "openvino/core/except.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/remote_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

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
             port.get_partial_shape().is_dynamic() ? ov::Shape{0} : port.get_partial_shape().to_shape(),
             allocator) {}

Tensor::Tensor(const ov::Output<const ov::Node>& port, void* host_ptr, const Strides& byte_strides)
    : Tensor(port.get_element_type(),
             port.get_partial_shape().is_dynamic() ? ov::Shape{0} : port.get_partial_shape().to_shape(),
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

}  // namespace ov
