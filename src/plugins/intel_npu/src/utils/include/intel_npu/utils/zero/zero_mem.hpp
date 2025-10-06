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
    /**
     * @brief Allocates a new memory region in the level zero context provided through init_structs.
     * @param init_structs Holder for the level zero structures.
     * @param bytes Size in bytes of the memory that must be allocated.
     * @param alignment Alignment needed for the memory; it must be a multiple of the standard page size
     * @param is_input Optimize reads from this buffer. Specific level zero flags will be used for allocation in case
     * the buffer is intended to be used as an input.
     */
    ZeroMem(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
            const size_t bytes,
            const size_t alignment,
            const bool is_input);

    /**
     * @brief Imports an already allocated memory in the level zero context provided through init_structs.
     * @param init_structs Holder for the level zero structures.
     * @param data Memory to be imported
     * @param bytes Size in bytes of the memory that must be allocated.
     * @param is_input Optimize reads from this buffer. Specific level zero flags will be used for allocation in case
     * the buffer is intended to be used as an input.
     * @param standard_allocation If a CPU standard allocation is shared it must be set to true. Otherwise it will try
     * to import DMA-BUF (on Linux) or WIN32 (on Windows) memory.
     */
    ZeroMem(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
            const void* data,
            const size_t bytes,
            const bool is_input,
            const bool standard_allocation);

    /**
     *@brief Return allocated memory
     */
    void* data();

    /**
     *@brief Return size of the allocated memory
     */
    size_t size();

    /**
     *@brief Return memory id of the allocated memory
     */
    uint64_t id();

    ~ZeroMem();

private:
    std::shared_ptr<ZeroInitStructsHolder> _init_structs;
    Logger _logger;

    void* _ptr = nullptr;
    size_t _size = 0;
    uint64_t _id = 0;
};

/**
 * @brief Default Zero Memory exception for the cases when we can not import a memory from the given tensor and must
 * fallback on allocating a new zero memory and do memcpy
 */
class ZeroMemException final : public std::runtime_error {
public:
    explicit ZeroMemException(const std::string& msg) : std::runtime_error(msg) {}
};
}  // namespace intel_npu
