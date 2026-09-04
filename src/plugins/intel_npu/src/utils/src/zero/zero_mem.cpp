// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_mem.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

namespace intel_npu {

// MemoryTracer implementation
MemoryTracer& MemoryTracer::get_instance() {
    static MemoryTracer instance;
    return instance;
}

MemoryTracer::MemoryTracer() {
    const char* trace_file_env = std::getenv("NPU_MEMORY_TRACE_FILE");
    if (trace_file_env && trace_file_env[0] != '\0') {
        set_trace_file(trace_file_env);
    }
}

MemoryTracer::~MemoryTracer() {
    if (trace_file_.is_open()) {
        trace_file_.close();
    }
}

void MemoryTracer::set_trace_file(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(trace_mutex_);
    if (!trace_file_.is_open()) {
        trace_file_.open(file_path, std::ios::out | std::ios::app);
        if (trace_file_.is_open()) {
            enabled_ = true;
            write_header();
        }
    }
}

void MemoryTracer::write_header() {
    trace_file_ << "timestamp,operation,size_bytes,purpose,name,memory_id,pointer,alignment,alloc_type,is_input\n";
    trace_file_.flush();
}

void MemoryTracer::log_allocation(uint64_t id, size_t size, memory_purpose purpose, const std::string& name,
                                 void* ptr, size_t alignment, const char* alloc_type, bool is_input) {
    if (!enabled_) return;
    
    std::lock_guard<std::mutex> lock(trace_mutex_);
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time_t);
#else
    localtime_r(&time_t, &tm_buf);
#endif
    
    trace_file_ << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S")
                << '.' << std::setfill('0') << std::setw(3) << ms.count()
                << ",alloc," << size
                << "," << memory_purpose_to_string(purpose)
                << ",\"" << (name.empty() ? "<unnamed>" : name) << "\""
                << "," << id
                << ",0x" << std::hex << reinterpret_cast<uintptr_t>(ptr) << std::dec
                << "," << alignment
                << "," << alloc_type
                << "," << (is_input ? "true" : "false")
                << "\n";
    trace_file_.flush();
}

void MemoryTracer::log_deallocation(uint64_t id, size_t size, memory_purpose purpose, const std::string& name,
                                   void* ptr, const char* alloc_type) {
    if (!enabled_) return;
    
    std::lock_guard<std::mutex> lock(trace_mutex_);
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time_t);
#else
    localtime_r(&time_t, &tm_buf);
#endif
    
    trace_file_ << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S")
                << '.' << std::setfill('0') << std::setw(3) << ms.count()
                << ",free," << size
                << "," << memory_purpose_to_string(purpose)
                << ",\"" << (name.empty() ? "<unnamed>" : name) << "\""
                << "," << id
                << ",0x" << std::hex << reinterpret_cast<uintptr_t>(ptr) << std::dec
                << ",0"  // alignment not relevant for free
                << "," << alloc_type
                << ",false"  // is_input not relevant for free
                << "\n";
    trace_file_.flush();
}

ZeroMem::ZeroMem(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                 const size_t bytes,
                 const size_t alignment,
                 const bool is_input,
                 memory_purpose purpose,
                 const std::string& name)
    : _init_structs(init_structs),
      _logger("ZeHostMem", Logger::global().level()),
      _size(bytes == 0 ? alignment : (bytes + alignment - 1) & ~(alignment - 1)),
      _alignment(alignment),
      _purpose(purpose),
      _name(name),
      _alloc_type("host") {
    uint32_t zero_memory_flag = 0;
    if (is_input) {
        zero_memory_flag = ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED;
    }

    ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, zero_memory_flag};
    THROW_ON_FAIL_FOR_LEVELZERO("zeMemAllocHost",
                                zeMemAllocHost(_init_structs->getContext(), &desc, _size, alignment, &_ptr));

    _id = zeroUtils::get_l0_context_memory_allocation_id(_init_structs->getContext(), _ptr);
    OPENVINO_ASSERT(_id != 0, "Failed to get memory allocation id of the allocated memory");
    
    // Log allocation
    MemoryTracer::get_instance().log_allocation(_id, _size, _purpose, _name, _ptr, _alignment, _alloc_type, is_input);
}

ZeroMem::ZeroMem(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                 const void* data,
                 const size_t bytes,
                 const bool is_input,
                 const bool standard_allocation,
                 memory_purpose purpose,
                 const std::string& name)
    : _init_structs(init_structs),
      _logger("ZeHostMem", Logger::global().level()),
      _size(bytes),
      _alignment(utils::STANDARD_PAGE_SIZE),
      _purpose(purpose),
      _name(name),
      _alloc_type(standard_allocation ? "standard_import" : "shared_import") {
    if (standard_allocation) {
        if (!_init_structs->isExternalMemoryStandardAllocationSupported()) {
            throw ZeroMemException("Importing standard allocation is not supported with this driver version");
        }

        if (!utils::memory_and_size_aligned_to_standard_page_size(data, _size)) {
            throw ZeroMemException(
                "Importing standard allocation is not supported if memory is not aligned to standard page size");
        }

        // Reject the import only when the region genuinely overlaps a previously imported
        // allocation. Probe the last valid byte (data + _size - 1) so that a buffer whose end
        // merely abuts an adjacent allocation is still importable. Other cases are handled by
        // the driver.
        if (_size > 0 && zeroUtils::get_l0_context_memory_allocation_id(
                             _init_structs->getContext(),
                             static_cast<void*>(static_cast<uint8_t*>(const_cast<void*>(data)) + _size - 1)) > 0) {
            throw ZeroMemException("Can not import a memory which is part of an existing allocation");
        }

        uint32_t zero_memory_flag = 0;
        if (is_input) {
            zero_memory_flag = ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED;
        }
        ze_external_memmap_sysmem_ext_desc_t memory_import = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMMAP_SYSMEM_EXT_DESC,
                                                              nullptr,
                                                              const_cast<void*>(data),
                                                              _size};
        ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, &memory_import, zero_memory_flag};
        auto result = zeMemAllocHost(_init_structs->getContext(), &desc, _size, utils::STANDARD_PAGE_SIZE, &_ptr);

        if (result != ZE_RESULT_SUCCESS) {
            throw ZeroMemException("Importing memory failed with result " + ze_result_to_string(result) + " - " +
                                   ze_result_to_description(result).c_str());
        }
    } else {
        OPENVINO_ASSERT(_init_structs->isExternalMemoryFdWin32Supported(),
                        "Remote tensor functionality is not supported with this driver version");

        OPENVINO_ASSERT(data != nullptr, "Data pointer for importing memory can't be null");
#ifdef _WIN32
        ze_external_memory_import_win32_handle_t memory_import = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32,
                                                                  nullptr,
                                                                  ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32,
                                                                  const_cast<void*>(data),
                                                                  nullptr};
#else
        ze_external_memory_import_fd_t memory_import = {ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD,
                                                        nullptr,
                                                        ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
                                                        static_cast<int>(reinterpret_cast<intptr_t>(data))};
#endif
        ze_host_mem_alloc_desc_t desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, &memory_import, 0};
        THROW_ON_FAIL_FOR_LEVELZERO(
            "zeMemAllocHost",
            zeMemAllocHost(_init_structs->getContext(), &desc, _size, utils::STANDARD_PAGE_SIZE, &_ptr));
    }

    _id = zeroUtils::get_l0_context_memory_allocation_id(_init_structs->getContext(), _ptr);
    OPENVINO_ASSERT(_id != 0, "Failed to get memory allocation id of the imported memory");
    
    // Log allocation
    MemoryTracer::get_instance().log_allocation(_id, _size, _purpose, _name, _ptr, _alignment, _alloc_type, is_input);
}

void* ZeroMem::data() {
    return _ptr;
}

size_t ZeroMem::size() {
    return _size;
}

uint64_t ZeroMem::id() {
    return _id;
}

ZeroMem::~ZeroMem() {
    // Log deallocation before freeing
    MemoryTracer::get_instance().log_deallocation(_id, _size, _purpose, _name, _ptr, _alloc_type);
    
    auto ze_context = _init_structs->getContext();
    if (ze_context == nullptr) {
        _logger.warning("Context is null while trying to free memory with id %llu. Memory might be already freed.",
                        _id);
        return;
    }

    auto result = zeMemFree(ze_context, _ptr);
    if (ZE_RESULT_SUCCESS != result) {
        _logger.error("L0 zeMemFree result: %s, code %#X - %s",
                      ze_result_to_string(result).c_str(),
                      uint64_t(result),
                      ze_result_to_description(result).c_str());
    }
}

}  // namespace intel_npu
