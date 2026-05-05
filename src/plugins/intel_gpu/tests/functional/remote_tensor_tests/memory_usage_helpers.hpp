// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Shared RAM/VRAM probing helpers for GPU remote-tensor functional tests.
//
// Provides:
//   * ov_test_memory::ProcessRamInfo / query_process_memory()  - cross-platform process RAM (Win + Linux)
//   * ov_test_memory::query_process_memory(PROCESS_MEMORY_COUNTERS_EX&) - Windows-only raw variant
//   * ov_test_memory::print_ram_info(label)                    - Windows-only RAM dump
//   * ov_test_memory::print_gpu_memory_info(label)             - Windows-only DXGI VRAM dump (auto-picks first HW adapter)
//   * ov_test_memory::print_gpu_memory_info(IDXGIAdapter1*, label) - Windows-only DXGI VRAM dump for a given adapter
//   * ov_test_memory::GpuMemoryInfo / query_vulkan_gpu_memory  - Vulkan VRAM probing (gated on prior <vulkan/vulkan.h>)
//   * ov_test_memory::bytes_to_mb(bytes)                       - byte->MB convenience

#pragma once

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#        define NOMINMAX_DEFINED_OV_TEST_MEMORY_USAGE
#    endif
#    include <windows.h>
#    include <atlbase.h>
#    include <dxgi1_4.h>
#    include <psapi.h>
#    ifdef NOMINMAX_DEFINED_OV_TEST_MEMORY_USAGE
#        undef NOMINMAX
#        undef NOMINMAX_DEFINED_OV_TEST_MEMORY_USAGE
#    endif
#elif defined(__linux__)
#    include <cstdio>
#    include <fstream>
#endif

namespace ov_test_memory {

inline double bytes_to_mb(uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

struct ProcessRamInfo {
    double working_set_mb = 0.0;
    double private_mb = 0.0;
    bool valid = false;
};

inline ProcessRamInfo query_process_memory() {
    ProcessRamInfo info;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                             sizeof(counters))) {
        info.working_set_mb = bytes_to_mb(counters.WorkingSetSize);
        info.private_mb = bytes_to_mb(counters.PrivateUsage);
        info.valid = true;
    }
#elif defined(__linux__)
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line)) {
        double kb = 0.0;
        if (line.rfind("VmRSS:", 0) == 0 && std::sscanf(line.c_str(), "VmRSS: %lf", &kb) == 1) {
            info.working_set_mb = kb / 1024.0;
            info.valid = true;
        } else if (line.rfind("VmSize:", 0) == 0 && std::sscanf(line.c_str(), "VmSize: %lf", &kb) == 1) {
            info.private_mb = kb / 1024.0;
        }
    }
#endif
    return info;
}

#ifdef _WIN32
inline bool query_process_memory(PROCESS_MEMORY_COUNTERS_EX& counters) {
    std::memset(&counters, 0, sizeof(counters));
    counters.cb = sizeof(counters);
    return GetProcessMemoryInfo(GetCurrentProcess(),
                                reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                                sizeof(counters)) == TRUE;
}

inline void print_ram_info(const std::string& label) {
    const auto info = query_process_memory();
    if (info.valid) {
        std::cout << "[INFO] RAM " << label
                  << ": working_set=" << info.working_set_mb << " MB"
                  << ", private=" << info.private_mb << " MB\n";
    } else {
        std::cout << "[INFO] RAM " << label << ": query failed\n";
    }
}

inline void print_gpu_memory_info(IDXGIAdapter1* adapter, const std::string& label) {
    if (!adapter) {
        std::cout << "[INFO] GPU memory " << label << ": null adapter\n";
        return;
    }
    IDXGIAdapter3* raw_adapter3 = nullptr;
    if (FAILED(adapter->QueryInterface(IID_PPV_ARGS(&raw_adapter3))) || !raw_adapter3) {
        std::cout << "[INFO] " << label << ": Failed to QI IDXGIAdapter3 for GPU memory query\n";
        return;
    }
    CComPtr<IDXGIAdapter3> adapter3(raw_adapter3);
    DXGI_QUERY_VIDEO_MEMORY_INFO local_info{}, non_local_info{};
    adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local_info);
    adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &non_local_info);
    std::cout << "[INFO] GPU memory " << label
              << ": local_used=" << bytes_to_mb(local_info.CurrentUsage) << " MB"
              << ", local_budget=" << bytes_to_mb(local_info.Budget) << " MB"
              << ", non_local_used=" << bytes_to_mb(non_local_info.CurrentUsage) << " MB"
              << ", non_local_budget=" << bytes_to_mb(non_local_info.Budget) << " MB\n";
}

inline void print_gpu_memory_info(const std::string& label) {
    IDXGIFactory4* raw_factory = nullptr;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&raw_factory))) || !raw_factory) {
        std::cout << "[INFO] GPU memory " << label << ": CreateDXGIFactory1 failed\n";
        return;
    }
    CComPtr<IDXGIFactory4> factory(raw_factory);
    UINT idx = 0;
    IDXGIAdapter1* raw_adapter = nullptr;
    while (factory->EnumAdapters1(idx++, &raw_adapter) != DXGI_ERROR_NOT_FOUND) {
        CComPtr<IDXGIAdapter1> adapter(raw_adapter);
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            continue;
        }
        print_gpu_memory_info(adapter, label);
        return;
    }
}
#endif  // _WIN32

#ifdef VULKAN_H_
struct GpuMemoryInfo {
    double used_mb = 0.0;
    double budget_mb = 0.0;
    bool valid = false;
};

namespace detail {
inline bool vk_has_device_extension(VkPhysicalDevice physical_device, const char* extension_name) {
    uint32_t extension_count = 0;
    if (vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr) != VK_SUCCESS) {
        return false;
    }
    std::vector<VkExtensionProperties> available_extensions(extension_count);
    if (vkEnumerateDeviceExtensionProperties(physical_device,
                                             nullptr,
                                             &extension_count,
                                             available_extensions.data()) != VK_SUCCESS) {
        return false;
    }
    for (const auto& ext : available_extensions) {
        if (std::strcmp(ext.extensionName, extension_name) == 0) {
            return true;
        }
    }
    return false;
}
}  // namespace detail

inline GpuMemoryInfo query_vulkan_gpu_memory(VkPhysicalDevice physical_device) {
    GpuMemoryInfo info;
#    ifdef VK_EXT_memory_budget
    if (!detail::vk_has_device_extension(physical_device, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
        return info;
    }

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget_properties{};
    budget_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;

    VkPhysicalDeviceMemoryProperties2 memory_properties{};
    memory_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    memory_properties.pNext = &budget_properties;
    vkGetPhysicalDeviceMemoryProperties2(physical_device, &memory_properties);

    uint64_t used_bytes = 0;
    uint64_t budget_bytes = 0;
    for (uint32_t i = 0; i < memory_properties.memoryProperties.memoryHeapCount; ++i) {
        const VkMemoryHeap& heap = memory_properties.memoryProperties.memoryHeaps[i];
        if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            used_bytes += budget_properties.heapUsage[i];
            budget_bytes += budget_properties.heapBudget[i];
        }
    }

    info.used_mb = bytes_to_mb(used_bytes);
    info.budget_mb = bytes_to_mb(budget_bytes);
    info.valid = budget_bytes > 0;
#    endif
    return info;
}
#endif  // VULKAN_H_

}  // namespace ov_test_memory
