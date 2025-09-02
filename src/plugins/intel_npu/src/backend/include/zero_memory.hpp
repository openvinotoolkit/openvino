// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <map>
#include <string>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/runtime/itensor.hpp"

namespace intel_npu {
namespace zeroMemory {

// Create an allocator that uses the ov::Allocator signature that will be used to create the tensor.
class HostMemAllocator {
public:
    explicit HostMemAllocator(const std::shared_ptr<ZeroInitStructsHolder>& initStructs, uint32_t flag = 0)
        : _initStructs(initStructs),
          _logger("HostMemAllocator", Logger::global().level()),
          _flag(flag) {}

    /**
     * @brief Allocates memory
     * @param bytes The size in bytes to allocate
     * @return Handle to the allocated resource
     */
    virtual void* allocate(const size_t bytes, const size_t alignment = utils::STANDARD_PAGE_SIZE) noexcept;

    /**
     * @brief Releases handle and all associated memory resources which invalidates the handle.
     * @param handle Pointer to allocated data
     * @return false if handle cannot be released, otherwise - true.
     */
    virtual bool deallocate(void* handle, const size_t bytes, size_t alignment = utils::STANDARD_PAGE_SIZE) noexcept;

    bool is_equal(const HostMemAllocator& other) const;

    virtual ~HostMemAllocator() = default;

protected:
    const std::shared_ptr<ZeroInitStructsHolder> _initStructs;

    Logger _logger;

    uint32_t _flag;
    static const std::size_t _alignment = utils::STANDARD_PAGE_SIZE;
};

class HostMemSharedAllocator final : public HostMemAllocator {
public:
    explicit HostMemSharedAllocator(const std::shared_ptr<ZeroInitStructsHolder>& initStructs,
                                    const std::shared_ptr<ov::ITensor>& tensor,
                                    uint32_t flag = 0)
        : HostMemAllocator(initStructs, flag),
          _tensor(tensor) {}

    void* allocate(const size_t bytes = 0, const size_t alignment = utils::STANDARD_PAGE_SIZE) noexcept override;

private:
    const std::shared_ptr<ov::ITensor> _tensor;
};

}  // namespace zeroMemory
}  // namespace intel_npu
