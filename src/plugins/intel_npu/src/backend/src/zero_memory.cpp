// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_memory.hpp"

#include "intel_npu/utils/zero/zero_api.hpp"
#include "zero_utils.hpp"

namespace intel_npu {
namespace zeroMemory {
DeviceMem::DeviceMem(const ze_device_handle_t device_handle, ze_context_handle_t context, const std::size_t size)
    : _size(size),
      _context(context),
      _log("DeviceMem", Logger::global().level()) {
    ze_device_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};

    zeroUtils::throwOnFail("zeMemAllocDevice",
                           zeMemAllocDevice(_context, &desc, _size, _alignment, device_handle, &_data));
}
DeviceMem& DeviceMem::operator=(DeviceMem&& other) {
    if (this == &other)
        return *this;
    free();
    _size = other._size;
    _data = other._data;
    _context = other._context;
    other._size = 0;
    other._data = nullptr;
    return *this;
}
void DeviceMem::free() {
    if (_size != 0) {
        _size = 0;
        zeroUtils::throwOnFail("zeMemFree DeviceMem", zeMemFree(_context, _data));
        _data = nullptr;
    }
}
DeviceMem::~DeviceMem() {
    try {
        free();
    } catch (const std::exception& e) {
        _log.error("Caught when freeing memory: %s", e.what());
    }
}

void* HostMemAllocator::allocate(const size_t bytes, const size_t /*alignment*/) noexcept {
    size_t size = bytes + _alignment - (bytes % _alignment);

    ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
                                     nullptr,
                                     static_cast<ze_host_mem_alloc_flags_t>(_flag)};
    void* data = nullptr;
    ze_result_t res = zeMemAllocHost(_initStructs->getContext(), &desc, size, _alignment, &data);

    if (res == ZE_RESULT_SUCCESS) {
        return data;
    } else {
        return nullptr;
    }
}
bool HostMemAllocator::deallocate(void* handle, const size_t /* bytes */, size_t /* alignment */) noexcept {
    auto result = zeMemFree(_initStructs->getContext(), handle);
    if (ZE_RESULT_SUCCESS != result) {
        return false;
    }

    return true;
}
bool HostMemAllocator::is_equal(const HostMemAllocator& other) const {
    return (_initStructs == other._initStructs) && (_flag == other._flag);
}

void MemoryManagementUnit::appendArgument(const std::string& name, const std::size_t argSize) {
    _offsets.emplace(std::make_pair(name, _size));

    _size += argSize + alignment -
             (argSize % alignment);  // is this really necessary? if 0==argSize%alignment -> add 1 * alignment
}

void MemoryManagementUnit::allocate(const ze_device_handle_t device_handle, const ze_context_handle_t context) {
    if (_size == 0) {
        OPENVINO_THROW("Can't allocate empty buffer");
    }

    _device = std::make_unique<DeviceMem>(device_handle, context, _size);
}
std::size_t MemoryManagementUnit::getSize() const {
    return _size;
}
const void* MemoryManagementUnit::getDeviceMemRegion() const {
    return _device ? _device->data() : nullptr;
}
void* MemoryManagementUnit::getDeviceMemRegion() {
    return _device ? _device->data() : nullptr;
}
void* MemoryManagementUnit::getDevicePtr(const std::string& name) {
    uint8_t* from = static_cast<uint8_t*>(_device ? _device->data() : nullptr);
    if (from == nullptr) {
        OPENVINO_THROW("Device memory not allocated yet");
    }
    if (!_offsets.count(name)) {
        OPENVINO_THROW("Invalid memory offset key: ", name);
    }

    return _offsets.at(name) + from;
}
}  // namespace zeroMemory
}  // namespace intel_npu
