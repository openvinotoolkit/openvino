// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_tensor.hpp"

#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_remote_tensor.hpp"
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
                       const bool isInput)
    : _init_structs(init_structs),
      _logger("ZeroTensor", config.get<LOG_LEVEL>()),
      _element_type{element_type},
      _shape{shape},
      _capacity{_shape},
      _strides{},
      _strides_once{} {
    OPENVINO_ASSERT(_element_type.is_static());
    if (isInput) {
        _zero_memory_flag = ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED;
    }
    const auto byte_size = ov::util::get_memory_size_safe(element_type, _shape);
    OPENVINO_ASSERT(byte_size, "Cannot allocate memory for type: ", element_type, " and shape: ", _shape);
    auto data = allocate_zero_memory(*byte_size, utils::STANDARD_PAGE_SIZE);
    OPENVINO_ASSERT(*byte_size == 0 || data != nullptr, "Failed to allocate zero memory");
    initialize_elements(data, element_type, _shape);
    _ptr = data;
}

ZeroTensor::ZeroTensor(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                       const ov::SoPtr<ov::ITensor>& user_tensor,
                       const Config& config)
    : _init_structs(init_structs),
      _logger("ZeroTensor", config.get<LOG_LEVEL>()),
      _element_type{user_tensor->get_element_type()},
      _shape{user_tensor->get_shape()},
      _capacity{_shape},
      _strides{user_tensor->get_strides()},
      _strides_once{},
      _imported_tensor(user_tensor) {
    OPENVINO_ASSERT(_element_type.is_static());

    auto remoteTensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(_imported_tensor._ptr);
    if (remoteTensor == nullptr) {
        if (zeroUtils::memory_was_allocated_in_the_same_l0_context(_init_structs->getContext(), user_tensor->data())) {
            _logger.debug("ZeroTensor::ZeroTensor - tensor was created in the same L0 context, size: %zu",
                          user_tensor->get_byte_size());

            _ptr = _imported_tensor->data();
        } else {
            OPENVINO_THROW("Tensor was not created in the same zero context");
        }
    } else {
        _ptr = remoteTensor->get_original_memory();
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
        deallocate_zero_memory(_ptr);
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
            OPENVINO_THROW("set_shape is not supported. Tensor re-allocation is not allowed for imported tensors.");
        }

        destroy_memory();

        // allocate buffer and initialize objects from scratch
        _capacity = _shape;
        _ptr = allocate_zero_memory(get_bytes_capacity(), utils::STANDARD_PAGE_SIZE);
        OPENVINO_ASSERT(get_bytes_capacity() == 0 || _ptr != nullptr, "Failed to allocate zero memory");
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

void* ZeroTensor::allocate_zero_memory(const size_t bytes, const size_t alignment) noexcept {
    size_t size = bytes + alignment - (bytes % alignment);

    ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, _zero_memory_flag};
    void* data = nullptr;
    auto result = zeMemAllocHost(_init_structs->getContext(), &desc, size, alignment, &data);

    if (result == ZE_RESULT_SUCCESS) {
        return data;
    } else {
        _logger.error("L0 zeMemAllocHost result: %s, code %#X - %s",
                      ze_result_to_string(result).c_str(),
                      uint64_t(result),
                      ze_result_to_description(result).c_str());
        return nullptr;
    }
}

void ZeroTensor::deallocate_zero_memory(void* handle) noexcept {
    auto result = zeMemFree(_init_structs->getContext(), handle);
    if (ZE_RESULT_SUCCESS != result) {
        _logger.error("L0 zeMemFree result: %s, code %#X - %s",
                      ze_result_to_string(result).c_str(),
                      uint64_t(result),
                      ze_result_to_description(result).c_str());
    }
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
