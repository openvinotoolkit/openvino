// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_remote_tensor.hpp"

#include <ze_api.h>

#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "openvino/core/memory_util.hpp"

using namespace ov::intel_npu;

namespace intel_npu {

ZeroRemoteTensor::ZeroRemoteTensor(const std::shared_ptr<ov::IRemoteContext>& context,
                                   const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                   const ov::element::Type& element_type,
                                   const ov::Shape& shape,
                                   TensorType tensor_type,
                                   MemType mem_type,
                                   const void* mem)
    : _context(context),
      _init_structs(init_structs),
      _element_type(element_type),
      _shape(shape),
      _capacity(shape),
      _logger("ZeroRemoteContext", Logger::global().level()),
      _tensor_type(tensor_type),
      _mem_type(mem_type),
      _mem(mem) {
    OPENVINO_ASSERT(shape_size(_shape) != 0);
    OPENVINO_ASSERT(_element_type.is_static());

    const auto byte_size = ov::util::get_memory_size_overflow(element_type, shape);
    OPENVINO_ASSERT(byte_size, "Cannot allocate memory for type: ", element_type, " and shape: ", shape);

    ze_device_external_memory_properties_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES;
    auto res = zeDeviceGetExternalMemoryProperties(_init_structs->getDevice(), &desc);
    if (res == ZE_RESULT_SUCCESS) {
#ifdef _WIN32
        if (desc.memoryAllocationImportTypes & ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32) {
            _external_memory_support = true;
        }
#else
        if (desc.memoryAllocationImportTypes & ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF) {
            _external_memory_support = true;
        }
#endif
    }

    allocate(*byte_size);
}

const ov::element::Type& ZeroRemoteTensor::get_element_type() const {
    return _element_type;
}

const ov::Shape& ZeroRemoteTensor::get_shape() const {
    return _shape;
}

const ov::Strides& ZeroRemoteTensor::get_strides() const {
    return _strides;
}

const ov::AnyMap& ZeroRemoteTensor::get_properties() const {
    return _properties;
}

void ZeroRemoteTensor::set_shape(ov::Shape new_shape) {
    if (_shape == new_shape) {
        return;
    }

    _shape = std::move(new_shape);

    if (ov::shape_size(_shape) > ov::shape_size(_capacity)) {
        OPENVINO_THROW("Cannot set a new bigger shape to this tensor.");
    }

    _strides.clear();
    update_strides();
}

void ZeroRemoteTensor::update_strides() {
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

const std::string& ZeroRemoteTensor::get_device_name() const {
    return _context->get_device_name();
}

std::shared_ptr<ov::IRemoteContext> ZeroRemoteTensor::get_context() const {
    return _context;
}

ZeroRemoteTensor::~ZeroRemoteTensor() {
    auto res = deallocate();
    if (!res) {
        _logger.error("ZeroRemoteTensor failed to free the memory");
    }
}

bool ZeroRemoteTensor::deallocate() noexcept {
    switch (_mem_type) {
    case MemType::L0_INTERNAL_BUF:
    case MemType::SHARED_BUF: {
        if (_data) {
            auto result = zeMemFree(_init_structs->getContext(), _data);
            if (ZE_RESULT_SUCCESS != result) {
                if (ZE_RESULT_ERROR_UNINITIALIZED == result) {
                    _logger.warning("ZeroRemoteTensor failed to free memory; Level zero context was already destroyed "
                                    "and memory was already released by the driver.");
                } else {
                    _logger.error("zeMemFree failed %#X", uint64_t(result));
                    return false;
                }
            }

            _data = nullptr;
        }

        return true;
    }
    default:
        return false;
    }
}

void ZeroRemoteTensor::allocate(const size_t bytes) {
    switch (_mem_type) {
    case MemType::L0_INTERNAL_BUF: {
        size_t size = (bytes + utils::STANDARD_PAGE_SIZE - 1) & ~(utils::STANDARD_PAGE_SIZE - 1);

        ze_host_mem_alloc_desc_t desc = {};
        if (_tensor_type == TensorType::INPUT) {
            ze_host_mem_alloc_flag_t flag = ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED;
            desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, static_cast<ze_host_mem_alloc_flags_t>(flag)};
        } else {
            desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
        }
        THROW_ON_FAIL_FOR_LEVELZERO(
            "zeMemAllocHost",
            zeMemAllocHost(_init_structs->getContext(), &desc, size, utils::STANDARD_PAGE_SIZE, &_data));
        break;
    }
    case MemType::SHARED_BUF: {
        if (!_external_memory_support) {
            OPENVINO_THROW("Remote tensor functionality is not supported with this driver version");
        }

        // set up the request to import the external memory handle
#ifdef _WIN32
        // in the case of the Windows platform memory is locked by the D3D12 memory management - using zeMemAllocDevice
        // to import memory
        ze_external_memory_import_win32_handle_t memory_import = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32,
                                                                  nullptr,
                                                                  ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32,
                                                                  const_cast<void*>(_mem),
                                                                  nullptr};
        ze_device_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, &memory_import, 0, 0};
        THROW_ON_FAIL_FOR_LEVELZERO("zeMemAllocDevice",
                                    zeMemAllocDevice(_init_structs->getContext(),
                                                     &desc,
                                                     bytes,
                                                     utils::STANDARD_PAGE_SIZE,
                                                     _init_structs->getDevice(),
                                                     &_data));
#else
        // in the case of Linux platforms memory could be changed after allocation - using zeMemAllocHost for importing
        // memory
        ze_external_memory_import_fd_t memory_import = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD,
                                                        nullptr,
                                                        ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
                                                        static_cast<int>(reinterpret_cast<intptr_t>(_mem))};
        ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, &memory_import, 0};
        THROW_ON_FAIL_FOR_LEVELZERO(
            "zeMemAllocHost",
            zeMemAllocHost(_init_structs->getContext(), &desc, bytes, utils::STANDARD_PAGE_SIZE, &_data));
#endif
        break;
    }
    default:
        _data = nullptr;
    }

    update_properties();
    update_strides();
}

bool ZeroRemoteTensor::is_allocated() const noexcept {
    return _data != nullptr;
}

void ZeroRemoteTensor::update_properties() {
    OPENVINO_ASSERT(is_allocated(), "Can't initialize ZeroRemoteTensor parameters as memory was not allocated");

    switch (_mem_type) {
    case MemType::L0_INTERNAL_BUF:
        _properties = {mem_type(_mem_type), mem_handle(_data), tensor_type(_tensor_type)};

        break;
    case MemType::SHARED_BUF:
        _properties = {mem_type(_mem_type), mem_handle(_data)};

        break;
    default:
        OPENVINO_THROW("Unsupported object type ", static_cast<int>(_mem_type));
    }
}

void* ZeroRemoteTensor::get_original_memory() const {
    return _data;
}

ze_context_handle_t ZeroRemoteTensor::get_zero_context_handle() const {
    return _init_structs->getContext();
}

}  // namespace intel_npu
