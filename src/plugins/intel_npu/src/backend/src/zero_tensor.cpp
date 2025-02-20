// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_tensor.hpp"

#include "intel_npu/config/common.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"

namespace intel_npu {

ZeroTensor::ZeroTensor(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                       const Config& config,
                       const ov::element::Type element_type,
                       const ov::Shape& shape,
                       const ov::Allocator& allocator)
    : _init_structs(init_structs),
      _logger("ZeroTensor", config.get<LOG_LEVEL>()),
      _element_type{element_type},
      _shape{shape},
      _capacity{_shape},
      _strides{},
      _strides_once{},
      _allocator{allocator} {
    OPENVINO_ASSERT(_element_type.is_static());
    OPENVINO_ASSERT(allocator, "Allocator was not initialized");
    const auto byte_size = ov::element::get_memory_size(_element_type, shape_size(_shape));
    auto data = const_cast<ov::Allocator&>(_allocator).allocate(byte_size);
    OPENVINO_ASSERT(byte_size == 0 || data != nullptr, "Failed to allocate memory");
    initialize_elements(data, element_type, _shape);
    _ptr = data;
}

void* ZeroTensor::data(const ov::element::Type& element_type) const {
    if (element_type != ov::element::dynamic &&
        (element_type.bitwidth() != get_element_type().bitwidth() ||
         element_type.is_real() != get_element_type().is_real() ||
         (element_type == ov::element::string && get_element_type() != ov::element::string) ||
         (element_type != ov::element::string && get_element_type() == ov::element::string))) {
        OPENVINO_THROW("Tensor data with element type ",
                       get_element_type(),
                       ", is not representable as pointer to ",
                       element_type);
    }
    return _ptr;
}

const ov::element::Type& ZeroTensor::get_element_type() const {
    return _element_type;
}

const ov::Shape& ZeroTensor::get_shape() const {
    return _shape;
}

void ZeroTensor::update_strides() const {
    if (_element_type.bitwidth() < 8) {
        return;
    }

    auto& shape = get_shape();
    if (_strides.empty() && !shape.empty()) {
        _strides.resize(shape.size());
        _strides.back() = shape.back() == 0 ? 0 : _element_type.size();
        std::transform(shape.crbegin(),
                       shape.crend() - 1,
                       _strides.rbegin(),
                       _strides.rbegin() + 1,
                       std::multiplies<size_t>());
    }
}

const ov::Strides& ZeroTensor::get_strides() const {
    OPENVINO_ASSERT(_element_type.bitwidth() >= 8,
                    "Could not get strides for types with bitwidths less then 8 bit. Tensor type: ",
                    _element_type);
    std::call_once(_strides_once, &ZeroTensor::update_strides, this);
    return _strides;
}

void ZeroTensor::initialize_elements(void* data, const ov::element::Type& element_type, const ov::Shape& shape) {
    if (element_type == ov::element::Type_t::string) {
        auto num_elements = shape_size(shape);
        auto string_ptr = static_cast<std::string*>(data);
        std::uninitialized_fill_n(string_ptr, num_elements, std::string());
    }
}

size_t ZeroTensor::get_capacity() const {
    return shape_size(_capacity);
}

size_t ZeroTensor::get_bytes_capacity() const {
    return ov::element::get_memory_size(get_element_type(), get_capacity());
}

void ZeroTensor::destroy_elements(size_t begin_ind, size_t end_ind) {
    // it removes elements from tail
    if (get_element_type() == ov::element::Type_t::string) {
        auto strings = static_cast<std::string*>(_ptr);
        for (size_t ind = begin_ind; ind < end_ind; ++ind) {
            using std::string;
            strings[ind].~string();
        }
    }
}

void ZeroTensor::destroy_memory() {
    destroy_elements(0, get_capacity());
    _allocator.deallocate(_ptr, get_bytes_capacity());
    _ptr = nullptr;
}

void ZeroTensor::set_shape(ov::Shape new_shape) {
    if (_shape == new_shape) {
        return;
    }

    _shape = std::move(new_shape);

    if (get_size() > get_capacity()) {
        if (_init_structs->getMutableCommandListExtVersion() < ZE_MAKE_VERSION(1, 0)) {
            OPENVINO_THROW("Re-shaping the tensor with a larger shape is not available using this driver version. "
                           "Please update the driver to the latest version.");
        }

        destroy_memory();

        // allocate buffer and initialize objects from scratch
        _capacity = _shape;
        _ptr = _allocator.allocate(get_bytes_capacity());
        initialize_elements(_ptr, _element_type, _shape);

        _reset_tensor_memory = true;
    }

    _strides.clear();
    update_strides();
}

bool ZeroTensor::memory_address_changed() {
    return _reset_tensor_memory;
}

void ZeroTensor::reset_memory_flag() {
    _reset_tensor_memory = false;
}

bool ZeroTensor::tensor_was_shared_with_user() {
    return _tensor_shared_with_user;
}
void ZeroTensor::set_tensor_shared_with_user() {
    _tensor_shared_with_user = true;
}

ZeroTensor::~ZeroTensor() {
    try {
        destroy_memory();
    } catch (const std::exception& ex) {
        _logger.error("Failed to destroy Zero Tensor: %s", ex.what());
    } catch (...) {
        _logger.error("Unexpected error when Zero Tensor is destroyed");
    }
}

}  // namespace intel_npu
