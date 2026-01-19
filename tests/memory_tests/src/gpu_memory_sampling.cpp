
#include <cstdio>

#include "gpu_memory_sampling.hpp"


#ifdef _WIN32

#include <windows.h>
#include <dxgi1_4.h>
#pragma comment(lib, "dxgi.lib")


static IDXGIFactory4 *dxgi_factory = nullptr;
static IDXGIAdapter3 *selected_adapter = nullptr;


int initGpuSampling() {
    if (FAILED(CreateDXGIFactory2(0, __uuidof(IDXGIFactory4), &(void *)dxgi_factory))) {
        fprintf(stderr, "CreateDXGIFactory2 failed\r\n");
        return -1;
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
        if (desc.VendorId == 0x8086) {
            fprintf(stderr, "Selecting GPU adapter #%d for sampling\r\n", adapterIndex);
            fprintf(stderr, "  description %ls\r\n", desc.Description);
            fprintf(stderr, "  vendor/device %x/%x\r\n", desc.VendorId, desc.DeviceId);
            fprintf(stderr, "  video mem:  %.2f MiB\r\n", desc.DedicatedVideoMemory / MiB);
            fprintf(stderr, "  system mem: %.2f MiB\r\n", desc.DedicatedSystemMemory / MiB);
            fprintf(stderr, "  shared mem: %.2f MiB\r\n", desc.SharedSystemMemory / MiB);
            fprintf(stderr, "\r\n");
            if (FAILED(adapter->QueryInterface(__uuidof(IDXGIAdapter3), &(void*)selected_adapter))) {
                fprintf(stderr, "Failed to query IDXGIAdapter3 interface for selected adapter\r\n");
                return -2;
            }
            adapter->Release();

            // try to take a sample?
            return 0;
        }
        adapter->Release();
    }

    fprintf(stderr, "No proper adapter was found for sampling\r\n");
    return -3;
}

GpuMemorySample sampleGpuMemory() {
    DXGI_ADAPTER_DESC desc;
    selected_adapter->GetDesc(&desc);
    int64_t total_mem = (desc.DedicatedVideoMemory
        + desc.DedicatedSystemMemory
        + desc.SharedSystemMemory) / 1024;
    int64_t budget = 0;
    int64_t used = 0;

    int nodeId = 0;
    DXGI_QUERY_VIDEO_MEMORY_INFO info;
    while (true) {
        auto result = selected_adapter->QueryVideoMemoryInfo(
            nodeId, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info);
        if (result != S_OK) {
            break;
        }
        budget += info.Budget / 1024;
        used += info.CurrentUsage / 1024;
        result = selected_adapter->QueryVideoMemoryInfo(
            nodeId, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info);
        if (result != S_OK) {
            break;
        }
        budget += info.Budget / 1024;
        used += info.CurrentUsage / 1024;
        nodeId += 1;
    }
    return {total_mem, budget, used};
}

#else

int initGpuSampling() {
    return -1;
}

GpuMemorySample sampleGpuMemory() {
    return {};
}

#endif
