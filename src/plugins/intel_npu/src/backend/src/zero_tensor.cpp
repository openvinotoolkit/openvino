// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_tensor.hpp"

#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_mem_pool.hpp"
#include "intel_npu/utils/zero/zero_remote_tensor.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/runtime/make_tensor.hpp"
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
      _strides{},
      _strides_once{},
      _is_input(is_input) {
    OPENVINO_ASSERT(_element_type.is_static());
    const auto byte_size = ov::util::get_memory_size_safe(element_type, _shape);
    OPENVINO_ASSERT(byte_size, "Cannot allocate memory for type: ", element_type, " and shape: ", _shape);

    _bytes_capacity = get_byte_size();

    _mem_ref = ZeroMemPool::get_instance().allocate_zero_memory(_init_structs,
                                                                byte_size.value(),
                                                                utils::STANDARD_PAGE_SIZE,
                                                                _is_input);
    auto data = _mem_ref->data();
    OPENVINO_ASSERT(byte_size.value() == 0 || data != nullptr, "Failed to allocate zero memory");
    _ptr = data;
    _can_be_reused = true;
}

ZeroTensor::ZeroTensor(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                       const Config& config,
                       const ov::SoPtr<ov::ITensor>& user_tensor)
    : _init_structs(init_structs),
      _logger("ZeroTensor", config.get<LOG_LEVEL>()),
      _user_tensor(user_tensor),
      _element_type{_user_tensor->get_element_type()},
      _shape{_user_tensor->get_shape()},
      _strides{_element_type.bitwidth() >= 8 ? _user_tensor->get_strides() : ov::Strides{}},
      _strides_once{} {
    OPENVINO_ASSERT(_element_type.is_static());

    _bytes_capacity = get_bytes_capacity();

    // Data pointer of the given user_tensor must be a valid address in the level zero context
    // Check first if the given tensor is a ov::IRemoteTensor (which has a different methods to expose the internal
    // storage)
    if (auto remote_tensor = std::dynamic_pointer_cast<ov::IRemoteTensor>(_user_tensor._ptr)) {
        if (auto zero_remote_tensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(remote_tensor)) {
            _ptr = zero_remote_tensor->get_original_memory();
        } else {
            std::optional<void*> mem_handle_object =
                zeroUtils::extract_object(remote_tensor->get_properties(), ov::intel_npu::mem_handle);
            OPENVINO_ASSERT(mem_handle_object.has_value(),
                            "Parameter with key ",
                            ov::intel_npu::mem_handle.name(),
                            " not found");
            _ptr = static_cast<uint8_t*>(mem_handle_object.value()) + ov::get_tensor_data_offset(*remote_tensor);
        }
    } else {
        _ptr = _user_tensor->data();
    }

    // Check if [data, data + size] was previously imported or allocated in the current level zero context. In such case
    // _mem_ref will keep a reference to that allocation. Otherwise the function will try to import it into the level
    // zero context.
    _logger.debug("ZeroTensor::ZeroTensor - get tensor from pool or import it");
    _mem_ref = ZeroMemPool::get_instance().import_standard_allocation_memory(_init_structs, _ptr, _bytes_capacity);
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

void* ZeroTensor::data_rw() {
    return data();
}

void* ZeroTensor::data_rw(const ov::element::Type& type) {
    return data(type);
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

size_t ZeroTensor::get_bytes_capacity() const {
    size_t original_shape_size = ov::shape_size(_shape);

    if (_user_tensor == nullptr || _element_type.bitwidth() < 8 || original_shape_size == 0 || _shape.empty() ||
        _strides.empty()) {
        return ov::util::get_memory_size(_element_type, original_shape_size);
    }

    return intel_npu::zeroUtils::get_capacity_size(_shape, _strides);
}

const ov::Strides& ZeroTensor::get_strides() const {
    OPENVINO_ASSERT(_element_type.bitwidth() >= 8,
                    "Could not get strides for types with bitwidths less than 8 bit. Tensor type: ",
                    _element_type);

    std::call_once(_strides_once, &ZeroTensor::update_strides, this);

    return _strides;
}

void ZeroTensor::set_shape(ov::Shape new_shape) {
    if (_shape == new_shape) {
        return;
    }

    _shape = std::move(new_shape);

    if (get_byte_size() > _bytes_capacity) {
        OPENVINO_ASSERT(_init_structs->getMutableCommandListExtVersion() >= ZE_MAKE_VERSION(1, 0),
                        "Re-shaping the tensor with a larger shape is not available using this driver version. "
                        "Please update the driver to the latest version.");

        OPENVINO_ASSERT(_user_tensor == nullptr,
                        "set_shape is not supported. Tensor re-allocation is not allowed for imported tensors.");

        _mem_ref.reset();
        _ptr = nullptr;

        // allocate buffer and initialize objects from scratch
        const auto byte_size = ov::util::get_memory_size_safe(_element_type, _shape);
        OPENVINO_ASSERT(byte_size, "Cannot allocate memory for type: ", _element_type, " and shape: ", _shape);
        _mem_ref = ZeroMemPool::get_instance().allocate_zero_memory(_init_structs,
                                                                    byte_size.value(),
                                                                    utils::STANDARD_PAGE_SIZE,
                                                                    _is_input);
        _ptr = _mem_ref->data();
        OPENVINO_ASSERT(byte_size.value() == 0 || _ptr != nullptr, "Failed to allocate zero memory");
        _bytes_capacity = get_byte_size();

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

ZeroTensor::~ZeroTensor() {
    _mem_ref = nullptr;  // Ensure that zero memory is destroyed before the user tensor is released
}

}  // namespace intel_npu
