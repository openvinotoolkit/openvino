// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_mem.hpp"

namespace intel_npu {

class ZeroMemPool final {
public:
    ZeroMemPool();
    ZeroMemPool(const ZeroMemPool& other) = delete;
    ZeroMemPool(ZeroMemPool&& other) = delete;
    void operator=(const ZeroMemPool&) = delete;
    void operator=(ZeroMemPool&&) = delete;

    static ZeroMemPool& get_instance();

    std::shared_ptr<ZeroMem> allocate_zero_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                                  const size_t bytes,
                                                  const size_t alignment,
                                                  const bool is_input = false);

    std::shared_ptr<ZeroMem> import_fd_win32_zero_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                                         const size_t bytes,
                                                         const size_t alignment,
                                                         const void* data);

    std::shared_ptr<ZeroMem> import_standard_allocation_zero_memory(
        const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
        const size_t bytes,
        const size_t alignment,
        const void* data,
        const bool is_input = false);

    std::shared_ptr<ZeroMem> get_zero_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                             const size_t bytes,
                                             const void* data);

private:
    void update_pool(ze_context_handle_t zero_context, const std::shared_ptr<intel_npu::ZeroMem>& zero_memory);
    void delete_pool_entry(ze_context_handle_t zero_context, ZeroMem* ptr);

    std::shared_ptr<ZeroInitStructsHolder> _init_structs = nullptr;

    std::unordered_map<uint64_t, std::weak_ptr<ZeroMem>> _pool;
    std::mutex _mutex;
};

}  // namespace intel_npu
