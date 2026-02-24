
#include <iostream>
#include "gpu_memory_sampling.hpp"

#define INTEL_PCI_VENDOR_ID 0x8086

#ifdef _WIN32

#include <windows.h>
#include <wrl/client.h>
#include <dxgi1_4.h>
#pragma comment(lib, "dxgi.lib")

namespace memory_tests::gpu {

using Microsoft::WRL::ComPtr;

static ComPtr<IDXGIFactory4> dxgi_factory;
static ComPtr<IDXGIAdapter3> selected_adapter;


InitStatus init() {
    if (S_OK != CreateDXGIFactory2(0, IID_PPV_ARGS(&dxgi_factory))) {
        std::cerr << "CreateDXGIFactory2 failed" << std::endl;
        return InitStatus::SUBSYSTEM_UNAVAILABLE;
    }
    IDXGIAdapter *adapter = nullptr;
    for (
        UINT adapterIndex = 0;
        dxgi_factory->EnumAdapters(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND;
        adapterIndex += 1
    ) {
        DXGI_ADAPTER_DESC desc;
        adapter->GetDesc(&desc);
        if (desc.VendorId == INTEL_PCI_VENDOR_ID) {
            // at this moment only looking for the first Intel GPU
            std::cerr << "Selecting adapter " << std::hex << desc.DeviceId << std::dec << std::endl;
            if (S_OK != adapter->QueryInterface(IID_PPV_ARGS(&selected_adapter))) {
                std::cerr << "Failed to query IDXGIAdapter3 interface for selected adapter" << std::endl;
                return InitStatus::SUBSYSTEM_UNSUPPORTED;
            }
            adapter->Release();
            return InitStatus::SUCCESS;
        }
        adapter->Release();
    }

    std::cerr << "No proper adapter was found for sampling" << std::endl;
    return InitStatus::GPU_NOT_FOUND;
}

Sample sample() {
    const int KiB = 1024;
    DXGI_ADAPTER_DESC desc;
    selected_adapter->GetDesc(&desc);

    Sample sample;
    sample.local_used = 0;
    sample.local_total = (desc.DedicatedVideoMemory + desc.DedicatedSystemMemory) / KiB;
    sample.nonlocal_used = 0;
    sample.nonlocal_total = desc.SharedSystemMemory / KiB;

    int nodeId = 0;
    DXGI_QUERY_VIDEO_MEMORY_INFO info;
    while (true) {
        auto result = selected_adapter->QueryVideoMemoryInfo(
            nodeId, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info);
        if (result != S_OK) {
            break;
        }
        sample.local_used += info.CurrentUsage / KiB;

        result = selected_adapter->QueryVideoMemoryInfo(
            nodeId, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info);
        if (result != S_OK) {
            break;
        }
        sample.nonlocal_used += info.CurrentUsage / KiB;
        nodeId += 1;
    }
    return sample;
}

}  // namespace memory_tests::gpu

#else

namespace memory_tests::gpu {

InitStatus init() {
    return InitStatus::SUBSYSTEM_UNAVAILABLE;
}

Sample sample() {
    return {};
}

}  // namespace memory_tests::gpu

#endif
