// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_tensor.hpp"

#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"

namespace intel_npu {

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

ZeroTensor::ZeroTensor(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                       const Config& config,
                       const ov::element::Type element_type,
                       const ov::Shape& shape,
                       const bool isInput,
                       const bool tensor_shared_with_user)
    : _init_structs(init_structs),
      _logger("ZeroTensor", config.get<LOG_LEVEL>()),
      _element_type{element_type},
      _shape{shape},
      _capacity{_shape},
      _strides{},
      _strides_once{},
      _tensor_shared_with_user{tensor_shared_with_user} {
    OPENVINO_ASSERT(_element_type.is_static());
    _allocator = isInput ? std::make_unique<zeroMemory::HostMemAllocator>(_init_structs,
                                                                          ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED)
                         : std::make_unique<zeroMemory::HostMemAllocator>(_init_structs);
    OPENVINO_ASSERT(_allocator, "Allocator was not initialized");
    const auto byte_size = ov::util::get_memory_size_safe(_element_type, _shape);
    OPENVINO_ASSERT(byte_size, "Cannot allocate memory for type: ", _element_type, " and shape: ", _shape);
    auto data = _allocator->allocate(*byte_size);
    OPENVINO_ASSERT(*byte_size == 0 || data != nullptr, "Failed to allocate memory");
    initialize_elements(data, _element_type, _shape);
    _ptr = data;
}

ZeroTensor::ZeroTensor(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                       const std::shared_ptr<ov::ITensor>& user_tensor,
                       const std::shared_ptr<ZeroTensor>& zero_tensor,
                       const Config& config,
                       const bool isInput,
                       const bool dynamic_batch_value_changed)
    : _init_structs(init_structs),
      _logger("ZeroTensor", config.get<LOG_LEVEL>()),
      _element_type{user_tensor->get_element_type()},
      _shape{user_tensor->get_shape()},
      _capacity{_shape},
      _strides{},
      _strides_once{} {
    OPENVINO_ASSERT(_element_type.is_static());

    if (zeroUtils::memory_was_allocated_in_the_same_l0_context(_init_structs->getContext(), user_tensor->data())) {
        _logger.debug("ZeroTensor::ZeroTensor - tensor was created in the same L0 context, size: %zu",
                      user_tensor->get_byte_size());

        _imported_tensor = user_tensor;
        _ptr = _imported_tensor->data();

        _tensor_shared_with_user = true;
        _update_command_list_arg = true;
    } else {
        if (dynamic_batch_value_changed || zero_tensor == nullptr || zero_tensor->tensor_was_shared_with_user()) {
            _logger.debug("ZeroTensor::ZeroTensor - create locally L0 tensor");
            _allocator =
                isInput ? std::make_unique<zeroMemory::HostMemAllocator>(_init_structs,
                                                                         ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED)
                        : std::make_unique<zeroMemory::HostMemAllocator>(_init_structs);
            OPENVINO_ASSERT(_allocator, "Allocator was not initialized");
            const auto byte_size = ov::util::get_memory_size_safe(_element_type, _shape);
            OPENVINO_ASSERT(byte_size, "Cannot allocate memory for type: ", _element_type, " and shape: ", _shape);
            auto data = _allocator->allocate(*byte_size);
            OPENVINO_ASSERT(*byte_size == 0 || data != nullptr, "Failed to allocate memory");
            initialize_elements(data, _element_type, _shape);

            _ptr = data;
            _update_command_list_arg = true;
        } else {
            _logger.debug("ZeroTensor::ZeroTensor - use existent level zero buffer");

            _imported_tensor = zero_tensor;
            _ptr = _imported_tensor->data();
        }
    }
}

ZeroTensor::ZeroTensor(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                       const std::shared_ptr<ZeroRemoteTensor>& zero_remote_tensor,
                       const Config& config,
                       const bool isInput)
    : _init_structs(init_structs),
      _logger("ZeroTensor", config.get<LOG_LEVEL>()),
      _element_type{zero_remote_tensor->get_element_type()},
      _shape{zero_remote_tensor->get_shape()},
      _capacity{_shape},
      _strides{},
      _strides_once{} {
    OPENVINO_ASSERT(_element_type.is_static());

    _imported_tensor = zero_remote_tensor;
    _ptr = zero_remote_tensor->get_original_memory();

    _tensor_shared_with_user = true;
    _update_command_list_arg = true;
}

ZeroTensor::ZeroTensor(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                       const std::shared_ptr<ov::ITensor>& user_tensor,
                       const Config& config)
    : _init_structs(init_structs),
      _logger("ZeroTensor", config.get<LOG_LEVEL>()),
      _element_type{user_tensor->get_element_type()},
      _shape{user_tensor->get_shape()},
      _capacity{_shape},
      _strides{},
      _strides_once{} {
    OPENVINO_ASSERT(_element_type.is_static());

    _imported_tensor = user_tensor;
    _ptr = user_tensor->data();

    _tensor_shared_with_user = true;
    _update_command_list_arg = true;
}

bool ZeroTensor::update_command_list_arg() {
    return _update_command_list_arg;
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
    return ov::util::get_memory_size(get_element_type(), get_capacity());
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
    if (_imported_tensor == nullptr) {
        destroy_elements(0, get_capacity());
        _allocator->deallocate(_ptr, get_bytes_capacity());
        _ptr = nullptr;
    }
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

        if (_imported_tensor != nullptr) {
            OPENVINO_THROW("Just throw it for now, no idea how this must work.");
        }

        destroy_memory();

        // allocate buffer and initialize objects from scratch
        _capacity = _shape;
        _ptr = _allocator->allocate(get_bytes_capacity());
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
