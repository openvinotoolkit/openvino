// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_remote_tensor.hpp"

#include <ze_api.h>

#include "intel_npu/config/common.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "openvino/core/type/element_iterator.hpp"

using namespace ov::intel_npu;

namespace {

constexpr std::size_t STANDARD_PAGE_SIZE = 4096;

}  // namespace

namespace intel_npu {

ZeroRemoteTensor::ZeroRemoteTensor(const std::shared_ptr<ov::IRemoteContext>& context,
                                   const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                   const ze_device_properties_t& device_properties,
                                   const ov::element::Type& element_type,
                                   const ov::Shape& shape,
                                   const Config& config,
                                   TensorType tensor_type,
                                   MemType mem_type,
                                   void* mem)
    : RemoteTensor(context, element_type, shape),
      _config(config),
      _logger("ZeroRemoteContext", _config.get<LOG_LEVEL>()),
      _init_structs(init_structs),
      _device_properties(device_properties),
      _tensor_type(tensor_type),
      _mem_type(mem_type),
      _mem(mem) {
    const auto byte_size = ov::element::get_memory_size(_element_type, shape_size(_shape));

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

    allocate(byte_size);
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
        size_t size = (bytes + STANDARD_PAGE_SIZE - 1) & ~(STANDARD_PAGE_SIZE - 1);

        ze_host_mem_alloc_desc_t desc = {};
        if (_tensor_type == TensorType::INPUT && (_device_properties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED)) {
            ze_host_mem_alloc_flag_t flag = ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED;
            desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, static_cast<ze_host_mem_alloc_flags_t>(flag)};
        } else {
            desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
        }
        THROW_ON_FAIL_FOR_LEVELZERO(
            "zeMemAllocHost",
            zeMemAllocHost(_init_structs->getContext(), &desc, size, STANDARD_PAGE_SIZE, &_data));
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
                                                                  _mem,
                                                                  nullptr};
        ze_device_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, &memory_import, 0, 0};
        THROW_ON_FAIL_FOR_LEVELZERO("zeMemAllocDevice",
                                    zeMemAllocDevice(_init_structs->getContext(),
                                                     &desc,
                                                     bytes,
                                                     STANDARD_PAGE_SIZE,
                                                     _init_structs->getDevice(),
                                                     &_data));
#else
        // in the case of Linux platforms memory could be changed after allocation - using zeMemAllocHost for importing
        // memory
        ze_external_memory_import_fd_t memory_import = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD,
                                                        nullptr,
                                                        ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
                                                        static_cast<int>(reinterpret_cast<intptr_t>(_mem))};
        ze_host_mem_alloc_desc_t desc = {.pNext = &memory_import};
        THROW_ON_FAIL_FOR_LEVELZERO(
            "zeMemAllocHost",
            zeMemAllocHost(_init_structs->getContext(), &desc, bytes, STANDARD_PAGE_SIZE, &_data));
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
