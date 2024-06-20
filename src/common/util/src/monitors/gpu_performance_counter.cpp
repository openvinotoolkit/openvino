// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/monitors/gpu_performance_counter.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "openvino/util/monitors/performance_counter.hpp"
#ifdef _WIN32
#    define NOMINMAX
#    include <dxgi.h>
#    include <pdh.h>
#    include <pdhmsg.h>
#    include <windows.h>

#    include <chrono>
#    include <string>
#    include <system_error>
#    include <thread>

#    include "query_wrapper.hpp"
#    define RENDER_ENGINE_COUNTER_INDEX  0
#    define COMPUTE_ENGINE_COUNTER_INDEX 1
#    define MAX_COUNTER_INDEX            2

namespace ov {
namespace util {
namespace monitor {
class GpuPerformanceCounter::PerformanceCounterImpl {
public:
    PerformanceCounterImpl() {
        auto devices = getNumberOfCores();
        int gpuIndex = 0;
        for (auto item : devices) {
            coreTimeCounters[item.first] = {};
            coreTimeCounters[item.first].resize(MAX_COUNTER_INDEX);
        }
        initCoreCounters(devices);
    }

    std::map<std::string, LUID> getNumberOfCores() {
        auto LuidToString = [](LUID luid) -> std::string {
            uint8_t highBytes[sizeof(luid.HighPart)] = {0};
            uint8_t lowBytes[sizeof(luid.LowPart)] = {0};
            std::stringstream ss;
            for (int index = 0; index < sizeof(luid.LowPart); index++) {
                lowBytes[sizeof(luid.LowPart) - 1 - index] = (luid.LowPart >> index * 8) & 0xFF;
                ss << std::hex << std::setw(2) << std::setfill('0')
                   << static_cast<int>(lowBytes[sizeof(luid.LowPart) - 1 - index]);
            }
            for (int index = 0; index < sizeof(luid.HighPart); index++) {
                highBytes[index] = (luid.HighPart >> index * 8) & 0xFF;
                ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(highBytes[index]);
            }
            return ss.str();
        };
        IDXGIFactory* pFactory;
        HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)(&pFactory));
        if (FAILED(hr))
            return {};
        int gpuIndex = 0;
        std::map<std::string, LUID> gpuCores;
        IDXGIAdapter* pAdapter;
        while (pFactory->EnumAdapters(gpuIndex, &pAdapter) != DXGI_ERROR_NOT_FOUND) {
            DXGI_ADAPTER_DESC desc;
            pAdapter->GetDesc(&desc);
            if (wcscmp(desc.Description, L"Microsoft Basic Render Driver") != 0 && desc.VendorId == 0x8086) {
                auto luidStr = LuidToString(desc.AdapterLuid);
                gpuCores[luidStr] = desc.AdapterLuid;
                wprintf(L"GPU Name: \n\t%sLUID: ", desc.Description);
                std::cout << "\t" << std::hex << desc.AdapterLuid.HighPart << "-" << desc.AdapterLuid.LowPart;
                std::cout << "\t LUID string: " << luidStr << std::endl;
            }
            gpuIndex++;
        }
        return gpuCores;
    }

    void initCoreCounters(const std::map<std::string, LUID>& devices) {
        auto LuidToString = [](LUID luid) -> std::string {
            std::stringstream ss;
            ss << std::hex << ((long long)luid.HighPart << 32 | luid.LowPart);
            return ss.str();
        };
        int gpuIndex = 0;
        for (auto item : devices) {
            std::string full3DCounterPath = std::string("\\GPU Engine(*_luid_*" + LuidToString(item.second) +
                                                        "_phys*engtype_3D)\\Utilization Percentage");
            std::string fullComputeCounterPath = std::string("\\GPU Engine(*_luid_*" + LuidToString(item.second) +
                                                             "_phys*engtype_Compute)\\Utilization Percentage");
            coreTimeCounters[item.first][RENDER_ENGINE_COUNTER_INDEX] =
                addCounter(query, expandWildCardPath(full3DCounterPath.c_str()));
            coreTimeCounters[item.first][COMPUTE_ENGINE_COUNTER_INDEX] =
                addCounter(query, expandWildCardPath(fullComputeCounterPath.c_str()));
        }
        auto status = PdhCollectQueryData(query);
    }

    std::vector<std::string> expandWildCardPath(LPCSTR WildCardPath) {
        PDH_STATUS Status = ERROR_SUCCESS;
        DWORD PathListLength = 0;
        DWORD PathListLengthBufLen;
        std::vector<std::string> pathList;
        Status = PdhExpandWildCardPathA(NULL, WildCardPath, NULL, &PathListLength, 0);
        if (Status != ERROR_SUCCESS && Status != PDH_MORE_DATA) {
            return pathList;
        }
        PathListLengthBufLen = PathListLength + 100;
        PZZSTR ExpandedPathList = (PZZSTR)malloc(PathListLengthBufLen);
        Status = PdhExpandWildCardPathA(NULL, WildCardPath, ExpandedPathList, &PathListLength, 0);
        if (Status != ERROR_SUCCESS) {
            free(ExpandedPathList);
            return pathList;
        }
        for (size_t i = 0; i < PathListLength;) {
            std::string path(ExpandedPathList + i);
            if (path.length() > 0) {
                // std::cout << path << std::endl;
                pathList.push_back(path);
                i += path.length() + 1;
            } else {
                break;
            }
        }
        free(ExpandedPathList);
        return pathList;
    }

    std::vector<PDH_HCOUNTER> addCounter(PDH_HQUERY Query, std::vector<std::string> pathList) {
        PDH_STATUS Status;
        std::vector<PDH_HCOUNTER> CounterList;
        for (std::string path : pathList) {
            PDH_HCOUNTER Counter;
            Status = PdhAddCounterA(Query, path.c_str(), NULL, &Counter);
            if (Status != ERROR_SUCCESS) {
                return CounterList;
            }
            Status = PdhSetCounterScaleFactor(Counter, -2);  // scale counter to [0, 1]
            if (ERROR_SUCCESS != Status) {
                return CounterList;
            }
            CounterList.push_back(Counter);
        }
        return CounterList;
    }

    std::map<std::string, double> get_load() {
        PDH_STATUS status;
        auto ts = std::chrono::system_clock::now();
        if (ts > lastTimeStamp) {
            auto delta =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastTimeStamp);
            if (delta.count() < 500) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500 - delta.count()));
            }
        }
        lastTimeStamp = std::chrono::system_clock::now();
        status = PdhCollectQueryData(query);
        PDH_FMT_COUNTERVALUE displayValue;
        std::map<std::string, double> gpuLoad;
        for (auto item : coreTimeCounters) {
            double value = 0;
            auto coreCounters = item.second;
            for (int counterIndex = 0; counterIndex < MAX_COUNTER_INDEX; counterIndex++) {
                auto countersList = coreCounters[counterIndex];
                for (auto counter : countersList) {
                    status = PdhGetFormattedCounterValue(counter, PDH_FMT_DOUBLE, NULL, &displayValue);
                    if (status != ERROR_SUCCESS) {
                        continue;
                    }
                    value += displayValue.doubleValue;
                }
            }
            gpuLoad[item.first] = value;
        }
        return gpuLoad;
    }

private:
    QueryWrapper query;
    std::map<std::string, std::vector<std::vector<PDH_HCOUNTER>>> coreTimeCounters;
    std::chrono::time_point<std::chrono::system_clock> lastTimeStamp = std::chrono::system_clock::now();
};

#elif defined(__linux__)
#    include <unistd.h>

#    include <chrono>
#    include <fstream>
#    include <regex>
#    include <utility>

namespace ov {
namespace util {
namespace monitor {
class GpuPerformanceCounter::PerformanceCounterImpl {
public:
    PerformanceCounterImpl() {}

    std::map<std::string, double> get_load() {
        // TODO: Implement.
        return {{"00000000", 0}};
    }
};

#else
namespace ov {
namespace util {
namespace monitor {
// not implemented
class GpuPerformanceCounter::PerformanceCounterImpl {
public:
    std::map<std::string, double> get_load() {
        return {{"00000000", 0}};
    };
#endif
GpuPerformanceCounter::GpuPerformanceCounter() : PerformanceCounter("GPU") {}
GpuPerformanceCounter::~GpuPerformanceCounter() {
    delete performanceCounter;
}
std::map<std::string, double> GpuPerformanceCounter::get_load() {
    if (!performanceCounter)
        performanceCounter = new PerformanceCounterImpl();
    return performanceCounter->get_load();
}
}
}
}