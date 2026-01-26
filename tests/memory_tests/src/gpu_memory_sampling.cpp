
#include <iostream>

#include "gpu_memory_sampling.hpp"


#ifdef _WIN32

#include <windows.h>
#include <dxgi1_4.h>
#pragma comment(lib, "dxgi.lib")

#define INTEL_VENDOR_ID 0x8086


static IDXGIFactory4 *dxgi_factory = nullptr;
static IDXGIAdapter3 *selected_adapter = nullptr;


std::ostream & operator<<(std::ostream &ostream, const DXGI_QUERY_VIDEO_MEMORY_INFO &info) {
    ostream << "{"
        << "\"Budget\": " << info.Budget
        << ", \"CurrentUsage\": " << info.CurrentUsage
        << ", \"AvailableForReservation\": " << info.AvailableForReservation
        << ", \"CurrentReservation\": " << info.CurrentReservation
    << "}";
}

std::ostream &operator<<(std::ostream &ostream, const DXGI_ADAPTER_DESC &desc) {
    ostream << "{"
        << "\"description\": \"" << desc.Description << "\""
        << ", \"vendor\": \"" << std::hex << desc.VendorId << "\""
        << ", \"device\": \"" << desc.DeviceId << std::dec << "\""
        << ", \"mem_video\": " << desc.DedicatedVideoMemory / MiB
        << ", \"mem_system\": " << desc.DedicatedSystemMemory / MiB
        << ", \"mem_shared\": " << desc.SharedSystemMemory / MiB
    << "}";
}


INIT_GPU_STATUS initGpuSampling() {
    if (S_OK != CreateDXGIFactory2(0, __uuidof(IDXGIFactory4), (void **)&dxgi_factory)) {
        std::cerr << "CreateDXGIFactory2 failed" << std::endl;
        return INIT_GPU_STATUS_SUBSYSTEM_UNAVAILABLE;
    }

    float MiB = 1024 * 1024;
    IDXGIAdapter *adapter = nullptr;
    for (
        UINT adapterIndex = 0;
        dxgi_factory->EnumAdapters(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND;
        adapterIndex += 1
    ) {
        DXGI_ADAPTER_DESC desc;
        adapter->GetDesc(&desc);
        if (desc.VendorId == INTEL_VENDOR_ID) {
            std::cerr << "Selecting adapter #" << adapterIndex << std::endl;
            std::cerr << desc << std::endl;
            if (S_OK != adapter->QueryInterface(__uuidof(IDXGIAdapter3), (void**)&selected_adapter)) {
                std::cerr << "Failed to query IDXGIAdapter3 interface for selected adapter" << std::endl;
                return INIT_GPU_STATUS_SUBSYSTEM_UNSUPPORTED;
            }
            adapter->Release();

            // try to take a sample?
            return INIT_GPU_STATUS_SUCCESS;
        }
        adapter->Release();
    }

    std::cerr << "No proper adapter was found for sampling" << std::endl;
    return INIT_GPU_STATUS_GPU_NOT_FOUND;
}

GpuMemorySample sampleGpuMemory() {
    DXGI_ADAPTER_DESC desc;
    selected_adapter->GetDesc(&desc);
    int64_t total_mem = (desc.DedicatedVideoMemory
        + desc.DedicatedSystemMemory
        + desc.SharedSystemMemory) / 1024;
    int64_t budget = 0;
    int64_t available = 0;
    int64_t used = 0;

    int nodeId = 0;
    DXGI_QUERY_VIDEO_MEMORY_INFO info;
    while (true) {
        auto result = selected_adapter->QueryVideoMemoryInfo(
            nodeId, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info);
        if (result != S_OK) {
            break;
        }
        std::cerr << "Sample of local memory: " << info << std::endl;
        budget += info.Budget / 1024;
        available += info.AvailableForReservation / 1024;
        used += info.CurrentUsage / 1024;

        result = selected_adapter->QueryVideoMemoryInfo(
            nodeId, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info);
        if (result != S_OK) {
            break;
        }
        std::cerr << "Sample of non-local memory: " << info << std::endl;
        budget += info.Budget / 1024;
        available += info.AvailableForReservation / 1024;
        used += info.CurrentUsage / 1024;
        nodeId += 1;
    }
    return {
        .total=total_mem,
        .free=available,
        .used=used
    };
}

#else

INIT_GPU_STATUS initGpuSampling() {
    return INIT_GPU_STATUS_SUBSYSTEM_UNAVAILABLE;
}

GpuMemorySample sampleGpuMemory() {
    return {};
}

#endif
