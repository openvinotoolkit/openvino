// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <map>
#include <string>

#include "intel_npu/utils/logger/logger.hpp"
#include "zero_init.hpp"

namespace {

constexpr std::size_t STANDARD_PAGE_SIZE = 4096;

}  // namespace

namespace intel_npu {
namespace zeroMemory {
struct DeviceMem {
    DeviceMem() = delete;
    DeviceMem(const ze_device_handle_t device_handle, const ze_context_handle_t context, const std::size_t size);
    DeviceMem(const DeviceMem&) = delete;
    DeviceMem(DeviceMem&& other)
        : _size(other._size),
          _data(other._data),
          _context(other._context),
          _log("DeviceMem", Logger::global().level()) {
        other._size = 0;
        other._data = nullptr;
    }
    DeviceMem& operator=(const DeviceMem&) = delete;
    DeviceMem& operator=(DeviceMem&& other);

    const void* data() const {
        return _data;
    }
    void* data() {
        return _data;
    }
    std::size_t size() const {
        return _size;
    }
    void free();
    ~DeviceMem();

private:
    std::size_t _size = 0;
    void* _data = nullptr;
    ze_context_handle_t _context = nullptr;
    static const std::size_t _alignment = STANDARD_PAGE_SIZE;

    Logger _log;
};

// Create an allocator that uses the ov::Allocator signature that will be used to create the tensor.
class HostMemAllocator final {
public:
    explicit HostMemAllocator(const std::shared_ptr<ZeroInitStructsHolder>& initStructs,
                              ze_host_mem_alloc_flag_t flag = {})
        : _initStructs(initStructs),
          _flag(flag) {}

    /**
     * @brief Allocates memory
     * @param bytes The size in bytes to allocate
     * @return Handle to the allocated resource
     */
    void* allocate(const size_t bytes, const size_t alignment = STANDARD_PAGE_SIZE) noexcept;

    /**
     * @brief Releases handle and all associated memory resources which invalidates the handle.
     * @param handle Pointer to allocated data
     * @return false if handle cannot be released, otherwise - true.
     */
    bool deallocate(void* handle, const size_t bytes, size_t alignment = STANDARD_PAGE_SIZE) noexcept;

    bool is_equal(const HostMemAllocator& other) const;

private:
    const std::shared_ptr<ZeroInitStructsHolder> _initStructs;

    ze_host_mem_alloc_flag_t _flag;
    static const std::size_t _alignment = STANDARD_PAGE_SIZE;
};

// Graph arguments (inputs and outputs) need to be allocated in the host memory.
// For discrete platforms, graph arguments need to be copied into the device memory.
// MemoryMangementUnit is used to allocate memory in the device memory.
// Usage: we should append graph arguments with corresponding names with `appendArgument` call to prepare size
// statistics and lookup table. To commit memory allocation we should call `allocate`
struct MemoryManagementUnit {
    MemoryManagementUnit() = default;

    void appendArgument(const std::string& name, const std::size_t argSize);

    void allocate(const ze_device_handle_t device_handle, const ze_context_handle_t context);

    std::size_t getSize() const;
    const void* getDeviceMemRegion() const;
    void* getDeviceMemRegion();

    void* getDevicePtr(const std::string& name);

    bool checkHostPtr(const void* ptr) const;

private:
    std::size_t _size = 0;

    std::unique_ptr<DeviceMem> _device;
    std::map<std::string, std::size_t> _offsets;

    static const std::size_t alignment = STANDARD_PAGE_SIZE;
};

}  // namespace zeroMemory
}  // namespace intel_npu
