// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"

namespace intel_npu {

struct ZeroMem final {
public:
    ZeroMem(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
            const size_t bytes,
            const size_t alignment,
            const uint32_t zero_memory_flag = 0,
            const void* data = nullptr,
            const bool import_fd_win32 = false);

    ~ZeroMem();

    void* _ptr = nullptr;
    size_t _size = 0;

private:
    std::shared_ptr<ZeroInitStructsHolder> _init_structs;

    Logger _logger;
};

class ZeroMemoryPool final {
public:
    ZeroMemoryPool();
    ZeroMemoryPool(const ZeroMemoryPool& other) = delete;
    ZeroMemoryPool(ZeroMemoryPool&& other) = delete;
    void operator=(const ZeroMemoryPool&) = delete;
    void operator=(ZeroMemoryPool&&) = delete;

    static ZeroMemoryPool& get_instance();

    std::shared_ptr<ZeroMem> allocate_and_get_zero_memory(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                                          const size_t bytes,
                                                          const size_t alignment,
                                                          const uint32_t zero_memory_flag = 0,
                                                          const void* data = nullptr,
                                                          const bool import_fd_win32 = false);

    std::shared_ptr<ZeroMem> get_zero_memory(const uint64_t id);

private:
    std::unordered_map<uint64_t, std::weak_ptr<ZeroMem>> _pool;
    std::mutex _mutex;
};

class ZeroTensorException final : public std::runtime_error {
public:
    explicit ZeroTensorException(const std::string& msg) : std::runtime_error(msg) {}
};

}
