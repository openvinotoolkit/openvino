// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <memory>
#include <mutex>
#include <string>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/zero/zero_mem_types.hpp"

namespace intel_npu {

class MemoryTracer {
public:
    static MemoryTracer& get_instance();
    void log_allocation(uint64_t id, size_t size, memory_purpose purpose, const std::string& name,
                       void* ptr, size_t alignment, const char* alloc_type, bool is_input);
    void log_deallocation(uint64_t id, size_t size, memory_purpose purpose, const std::string& name,
                         void* ptr, const char* alloc_type);
    void set_trace_file(const std::string& file_path);
private:
    MemoryTracer();
    ~MemoryTracer();
    void write_header();
    
    std::mutex trace_mutex_;
    std::ofstream trace_file_;
    bool enabled_ = false;
};

namespace zero_mem {
class ZeroMemPoolManager;
}  // namespace zero_mem

class ZeroMem final {
public:
    ZeroMem() = delete;

    /**
     * @brief Return allocated memory
     */
    void* data();

    /**
     * @brief Return size of the allocated memory
     */
    size_t size();

private:
    friend class zero_mem::ZeroMemPoolManager;

    /**
     * @brief Allocates a new memory region in the level zero context provided through init_structs.
     * @param init_structs Holder for the level zero structures.
     * @param bytes Size in bytes of the memory that must be allocated.
     * @param alignment Alignment needed for the memory; it must be a multiple of the standard page size
     * @param is_input Optimize reads from this buffer. Specific level zero flags will be used for allocation in case
     * the buffer is intended to be used as an input.
     * @param purpose Purpose of this memory allocation (for tracking/analysis)
     * @param name Optional descriptive name for this allocation
     */
    ZeroMem(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
            const size_t bytes,
            const size_t alignment,
            const bool is_input,
            memory_purpose purpose = memory_purpose::unknown,
            const std::string& name = "");

    /**
     * @brief Imports an already allocated memory in the level zero context provided through init_structs.
     * @param init_structs Holder for the level zero structures.
     * @param data Memory to be imported
     * @param bytes Size in bytes of the memory that must be allocated.
     * @param is_input Optimize reads from this buffer. Specific level zero flags will be used for allocation in case
     * the buffer is intended to be used as an input.
     * @param standard_allocation If a CPU standard allocation is shared it must be set to true. Otherwise it will try
     * to import DMA-BUF (on Linux) or WIN32 (on Windows) memory.
     * @param purpose Purpose of this memory allocation (for tracking/analysis)
     * @param name Optional descriptive name for this allocation
     */
    ZeroMem(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
            const void* data,
            const size_t bytes,
            const bool is_input,
            const bool standard_allocation,
            memory_purpose purpose = memory_purpose::unknown,
            const std::string& name = "");

    /**
     * @brief Return memory id of the allocated memory
     */
    uint64_t id();

    ~ZeroMem();

    std::shared_ptr<ZeroInitStructsHolder> _init_structs;
    Logger _logger;

    void* _ptr = nullptr;
    size_t _size = 0;
    uint64_t _id = 0;
    size_t _alignment = 0;
    memory_purpose _purpose = memory_purpose::unknown;
    std::string _name;
    const char* _alloc_type = "unknown";
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
