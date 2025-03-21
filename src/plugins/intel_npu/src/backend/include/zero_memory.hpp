// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <map>
#include <string>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"

namespace {

constexpr std::size_t STANDARD_PAGE_SIZE = 4096;

}  // namespace

namespace intel_npu {
namespace zeroMemory {

// Create an allocator that uses the ov::Allocator signature that will be used to create the tensor.
class HostMemAllocator final {
public:
    explicit HostMemAllocator(const std::shared_ptr<ZeroInitStructsHolder>& initStructs,
                              ze_host_mem_alloc_flag_t flag = {})
        : _initStructs(initStructs),
          _logger("HostMemAllocator", Logger::global().level()),
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

    Logger _logger;

    ze_host_mem_alloc_flag_t _flag;
    static const std::size_t _alignment = STANDARD_PAGE_SIZE;
};

}  // namespace zeroMemory
}  // namespace intel_npu
