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
            const bool is_input);

    ZeroMem(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
            const size_t bytes,
            const size_t alignment,
            const void* data);

    ZeroMem(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
            const size_t bytes,
            const size_t alignment,
            const void* data,
            const bool is_input);

    ~ZeroMem();

    void* _ptr = nullptr;
    size_t _size = 0;

private:
    std::shared_ptr<ZeroInitStructsHolder> _init_structs;
    Logger _logger;
};

class ZeroMemException final : public std::runtime_error {
public:
    explicit ZeroMemException(const std::string& msg) : std::runtime_error(msg) {}
};
}
