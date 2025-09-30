// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_tensor.hpp"

#include <ze_mem_import_system_memory_ext.h>

#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_host_tensor.hpp"
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
      _config(config),
      _logger("ZeroTensor", _config.get<LOG_LEVEL>()),
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
    _host_memory = ZeroMemoryPool::get_instance().allocate_and_get_zero_memory(_init_structs,
                                                                               _config,
                                                                               *byte_size,
                                                                               utils::STANDARD_PAGE_SIZE,
                                                                               _zero_memory_flag);
    auto data = _host_memory->_ptr;
    OPENVINO_ASSERT(*byte_size == 0 || data != nullptr, "Failed to allocate zero memory");
    _ptr = data;
    _can_be_reused = true;
}

ZeroTensor::ZeroTensor(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                       const ov::SoPtr<ov::ITensor>& user_tensor,
                       const Config& config)
    : _init_structs(init_structs),
      _config(config),
      _logger("ZeroTensor", _config.get<LOG_LEVEL>()),
      _element_type{user_tensor->get_element_type()},
      _shape{user_tensor->get_shape()},
      _capacity{_shape},
      _strides{_element_type.bitwidth() >= 8 ? user_tensor->get_strides() : ov::Strides{}},
      _strides_once{},
      _user_tensor(user_tensor) {
    OPENVINO_ASSERT(_element_type.is_static());

    // Data pointer of the given user_tensor must be a valid address in the level zero context
    // Check first if the given tensor is a ZeroRemoteTensor (which has a different method to expose the internal
    // storage) or ZeroHostTensor (it is a wrapper over ZeroRemoteTensor)
    auto remote_tensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(_user_tensor._ptr);
    auto host_tensor = std::dynamic_pointer_cast<ZeroHostTensor>(_user_tensor._ptr);
    if (remote_tensor == nullptr && host_tensor == nullptr) {
        // case for regular user tensor
        auto memory_id =
            zeroUtils::get_l0_context_memory_allocation_id(_init_structs->getContext(), _user_tensor->data());
        if (memory_id > 0) {
            _logger.debug("ZeroTensor::ZeroTensor - tensor was created in the same L0 context");
            _imported_tensor = _user_tensor;
            _host_memory = ZeroMemoryPool::get_instance().get_zero_memory(memory_id);
            if (_host_memory == nullptr) {
                throw ZeroTensorException("Failed to get zero memory from pool");
            }

            if (_host_memory->_size < _user_tensor->get_byte_size()) {
                throw ZeroTensorException("Imported memory size is smaller than tensor byte size");
            }

            if (_host_memory->_ptr != _user_tensor->data()) {
                auto user_ptr = reinterpret_cast<uintptr_t>(_user_tensor->data());
                auto host_ptr = reinterpret_cast<uintptr_t>(_host_memory->_ptr);
                auto offset = host_ptr - user_ptr;

                size_t tensor_byte_size = _user_tensor->get_byte_size();

                if (offset >= 0) {
                    if (tensor_byte_size > _host_memory->_size - ::abs(static_cast<std::ptrdiff_t>(offset))) {
                        throw ZeroTensorException("Tensor is out of bounds of the already allocated memory");
                    }
                } else {
                    OPENVINO_THROW("Tensor pointer is not part of the already allocated memory");
                }
            }

            _ptr = _user_tensor->data();
        } else if (_init_structs->isExternalMemoryStandardAllocationSupported() &&
                   utils::memory_and_size_aligned_to_standard_page_size(_user_tensor->data(),
                                                                        _user_tensor->get_byte_size())) {
            auto overlapping_end_address_check = zeroUtils::get_l0_context_memory_allocation_id(
                _init_structs->getContext(),
                static_cast<void*>(static_cast<uint8_t*>(_user_tensor->data()) + _user_tensor->get_byte_size()));

            if (overlapping_end_address_check > 0) {
                throw ZeroTensorException("Can not import a memory which is part of an existing allocation");
            }

            _logger.debug("ZeroTensor::ZeroTensor - import memory from a system memory pointer");

            _host_memory = ZeroMemoryPool::get_instance().allocate_and_get_zero_memory(_init_structs,
                                                                                       _config,
                                                                                       _user_tensor->get_byte_size(),
                                                                                       utils::STANDARD_PAGE_SIZE,
                                                                                       /*zero_memory_flag = */ 0,
                                                                                       _user_tensor->data());

            _ptr = _host_memory->_ptr;
        } else {
            throw ZeroTensorException("Tensor was not created in the same zero context");
        }
    } else {
        // case for ZeroRemoteTensors and ZeroHostTensors
        _imported_tensor = _user_tensor;
        if (host_tensor != nullptr) {
            remote_tensor = host_tensor->get_impl();
        }

        if (zeroUtils::get_l0_context_memory_allocation_id(_init_structs->getContext(),
                                                           remote_tensor->get_original_memory()) > 0) {
            _logger.debug("ZeroTensor::ZeroTensor - remote tensor was created in the same L0 context");
            _ptr = remote_tensor->get_original_memory();
        } else {
            throw ZeroTensorException("Tensor was not created in the same zero context");
        }
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
        if (_init_structs->getMutableCommandListExtVersion() < ZE_MAKE_VERSION(1, 0)) {
            OPENVINO_THROW("Re-shaping the tensor with a larger shape is not available using this driver version. "
                           "Please update the driver to the latest version.");
        }

        if (_imported_tensor != nullptr) {
            OPENVINO_THROW("set_shape is not supported. Tensor re-allocation is not allowed for imported tensors.");
        }

        _host_memory.reset();
        _ptr = nullptr;

        // allocate buffer and initialize objects from scratch
        _capacity = _shape;
        _host_memory = ZeroMemoryPool::get_instance().allocate_and_get_zero_memory(_init_structs,
                                                                                   _config,
                                                                                   get_bytes_capacity(),
                                                                                   utils::STANDARD_PAGE_SIZE,
                                                                                   _zero_memory_flag);
        _ptr = _host_memory->_ptr;
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

ZeHostMem::ZeHostMem(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                     const Config& config,
                     const size_t bytes,
                     const size_t alignment,
                     const uint32_t zero_memory_flag,
                     void* data)
    : _init_structs(init_structs),
      _logger("ZeHostMem", config.get<LOG_LEVEL>()) {
    ze_result_t result;
    if (data == nullptr) {
        _size = bytes + alignment - (bytes % alignment);
        ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, zero_memory_flag};
        result = zeMemAllocHost(_init_structs->getContext(), &desc, _size, alignment, &_ptr);
    } else {
        _size = bytes;
        _ze_external_memory_import_system_memory_t memory_import = {
            ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_SYSTEM_MEMORY,
            nullptr,
            data,
            _size};
        ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, &memory_import, zero_memory_flag};
        result = zeMemAllocHost(_init_structs->getContext(), &desc, _size, alignment, &_ptr);
    }

    if (result != ZE_RESULT_SUCCESS) {
        if (data == nullptr) {
            OPENVINO_THROW("L0 zeMemAllocHost result: ", ze_result_to_string(result), ", code ", uint64_t(result));
        } else {
            _logger.info("Importing memory through zeMemAllocHost failed, result: %s, code %#X - %s",
                         ze_result_to_string(result).c_str(),
                         uint64_t(result),
                         ze_result_to_description(result).c_str());

            throw ZeroTensorException("Importing memory failed");
        }
    }
}

ZeHostMem::~ZeHostMem() {
    auto result = zeMemFree(_init_structs->getContext(), _ptr);
    if (ZE_RESULT_SUCCESS != result) {
        _logger.error("L0 zeMemFree result: %s, code %#X - %s",
                      ze_result_to_string(result).c_str(),
                      uint64_t(result),
                      ze_result_to_description(result).c_str());
    }
}

ZeroMemoryPool::ZeroMemoryPool() {}

ZeroMemoryPool& ZeroMemoryPool::get_instance() {
    static ZeroMemoryPool instance;
    return instance;
}

std::shared_ptr<ZeHostMem> ZeroMemoryPool::allocate_and_get_zero_memory(
    const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
    const Config& config,
    const size_t bytes,
    const size_t alignment,
    const uint32_t zero_memory_flag,
    void* data) {
    auto zero_memory = std::shared_ptr<ZeHostMem>(
        new ZeHostMem(init_structs, config, bytes, alignment, zero_memory_flag, data),
        [this, zero_context = init_structs->getContext()](ZeHostMem* ptr) {
            auto memory_id = zeroUtils::get_l0_context_memory_allocation_id(zero_context, ptr->_ptr);

            std::lock_guard<std::mutex> lock(_mutex);
            if (_pool.at(memory_id).lock()) {
                // Don't destroy the command queue in case the shared ptr is in use!
                return;
            }
            _pool.erase(memory_id);
            // Destroy Command Queue
            delete ptr;
        });

    auto memory_id = zeroUtils::get_l0_context_memory_allocation_id(init_structs->getContext(), zero_memory->_ptr);
    OPENVINO_ASSERT(memory_id != 0, "Failed to get memory allocation id");

    auto pair = std::make_pair(memory_id, zero_memory);

    std::lock_guard<std::mutex> lock(_mutex);
    _pool.emplace(pair);

    return zero_memory;
}

std::shared_ptr<ZeHostMem> ZeroMemoryPool::get_zero_memory(const uint64_t id) {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_pool.find(id) != _pool.end()) {
        // found one weak pointer in the pool
        // is it valid?
        auto obj = _pool.at(id).lock();
        if (obj) {
            return obj;
        }
    }

    return nullptr;
}

}  // namespace intel_npu
