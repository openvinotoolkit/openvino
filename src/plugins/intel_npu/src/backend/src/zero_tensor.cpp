// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_tensor.hpp"

#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_mem_pool.hpp"
#include "intel_npu/utils/zero/zero_remote_tensor.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"

namespace {
bool is_pointer_representable(const ov::element::Type& tensor_type, const ov::element::Type& type) {
    if (type == ov::element::dynamic) {
        return true;
    } else {
        return (type.bitwidth() == tensor_type.bitwidth() && type.is_real() == tensor_type.is_real() &&
                type != ov::element::string && tensor_type != ov::element::string) ||
               (type == ov::element::string && tensor_type == ov::element::string);
    }
}
}  // namespace

namespace intel_npu {

ZeroTensor::ZeroTensor(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                       const Config& config,
                       const ov::element::Type element_type,
                       const ov::Shape& shape,
                       const bool is_input)
    : _init_structs(init_structs),
      _logger("ZeroTensor", config.get<LOG_LEVEL>()),
      _element_type{element_type},
      _shape{shape},
      _capacity{_shape},
      _strides{},
      _strides_once{},
      _is_input(is_input) {
    OPENVINO_ASSERT(_element_type.is_static());
    const auto byte_size = ov::util::get_memory_size_safe(element_type, _shape);
    OPENVINO_ASSERT(byte_size || byte_size.value() == 0,
                    "Cannot allocate memory for type: ",
                    element_type,
                    " and shape: ",
                    _shape);
    _mem_ref = ZeroMemPool::get_instance().allocate_zero_memory(_init_structs,
                                                                byte_size.value(),
                                                                utils::STANDARD_PAGE_SIZE,
                                                                _is_input);
    auto data = _mem_ref->data();
    OPENVINO_ASSERT(data != nullptr, "Failed to allocate zero memory");
    _ptr = data;
    _can_be_reused = true;
}

ZeroTensor::ZeroTensor(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                       const Config& config,
                       const ov::SoPtr<ov::ITensor>& user_tensor)
    : _init_structs(init_structs),
      _logger("ZeroTensor", config.get<LOG_LEVEL>()),
      _element_type{user_tensor->get_element_type()},
      _shape{user_tensor->get_shape()},
      _capacity{_shape},
      _strides{_element_type.bitwidth() >= 8 ? user_tensor->get_strides() : ov::Strides{}},
      _strides_once{},
      _user_tensor(user_tensor) {
    OPENVINO_ASSERT(_element_type.is_static());

    // Data pointer of the given user_tensor must be a valid address in the level zero context
    // Check first if the given tensor is a ZeroRemoteTensor (which has a different method to expose the internal
    // storage)
    auto remote_tensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(_user_tensor._ptr);
    if (remote_tensor == nullptr) {
        _ptr = _user_tensor->data();
    } else {
        _ptr = remote_tensor->get_original_memory();
    }

    // Check if [data, data + size] was previously imported or allocated in the current level zero context. In such case
    // _mem_ref will keep a reference to that allocation. Otherwise the function returns nullptr.
    _logger.debug("ZeroTensor::ZeroTensor - try to get tensor from pool");
    _mem_ref = ZeroMemPool::get_instance().get_zero_memory(_init_structs, _ptr, _user_tensor->get_byte_size());

    if (_mem_ref == nullptr) {
        _logger.debug("ZeroTensor::ZeroTensor - try to import memory from a system memory pointer");
        _mem_ref = ZeroMemPool::get_instance().import_standard_allocation_memory(_init_structs,
                                                                                 _ptr,
                                                                                 _user_tensor->get_byte_size());
    }
}

// Note: Override data() members to not used OpenVINO library code to improve performance
void* ZeroTensor::data() {
    return _ptr;
}

void* ZeroTensor::data(const ov::element::Type& type) {
    OPENVINO_ASSERT(is_pointer_representable(get_element_type(), type),
                    "Tensor data with element type ",
                    get_element_type(),
                    ", is not representable as pointer to ",
                    type);
    return data();
}

const void* ZeroTensor::data() const {
    return _ptr;
}

const void* ZeroTensor::data(const ov::element::Type& type) const {
    OPENVINO_ASSERT(is_pointer_representable(get_element_type(), type),
                    "Tensor data with element type ",
                    get_element_type(),
                    ", is not representable as pointer to ",
                    type);
    return data();
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
                    "Could not get strides for types with bitwidths less than 8 bit. Tensor type: ",
                    _element_type);
    std::call_once(_strides_once, &ZeroTensor::update_strides, this);
    return _strides;
}

size_t ZeroTensor::get_capacity() const {
    return shape_size(_capacity);
}

size_t ZeroTensor::get_bytes_capacity() const {
    return ov::util::get_memory_size(get_element_type(), get_capacity());
}

void ZeroTensor::set_shape(ov::Shape new_shape) {
    if (_shape == new_shape) {
        return;
    }

    _shape = std::move(new_shape);

    if (get_size() > get_capacity()) {
        OPENVINO_ASSERT(_init_structs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0),
                        "Re-shaping the tensor with a larger shape is not available using this driver version. "
                        "Please update the driver to the latest version.");

        OPENVINO_ASSERT(_user_tensor == nullptr,
                        "set_shape is not supported. Tensor re-allocation is not allowed for imported tensors.");

        _mem_ref.reset();
        _ptr = nullptr;

        // allocate buffer and initialize objects from scratch
        _capacity = _shape;
        _mem_ref = ZeroMemPool::get_instance().allocate_zero_memory(_init_structs,
                                                                    get_bytes_capacity(),
                                                                    utils::STANDARD_PAGE_SIZE,
                                                                    _is_input);
        _ptr = _mem_ref->data();
        OPENVINO_ASSERT(get_bytes_capacity() == 0 || _ptr != nullptr, "Failed to allocate zero memory");

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

void ZeroTensor::prevent_reuse() {
    _can_be_reused = false;
}

bool ZeroTensor::can_be_reused() {
    return _can_be_reused;
}

}  // namespace intel_npu
