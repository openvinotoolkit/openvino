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

#    include "openvino/util/wstring_convert_util.hpp"
#    include "query_wrapper.hpp"
#    define RENDER_ENGINE_COUNTER_INDEX  0
#    define COMPUTE_ENGINE_COUNTER_INDEX 1
#    define MAX_COUNTER_INDEX            2

namespace ov {
namespace util {
namespace monitor {
class GpuPerformanceCounter::PerformanceCounterImpl {
public:
    PerformanceCounterImpl(const std::string& deviceLuid) {
        luid = deviceLuid;
        if (!luid.empty()) {
            coreTimeCounters[luid] = {};
            coreTimeCounters[luid].resize(MAX_COUNTER_INDEX);
        }
        initCoreCounters(deviceLuid);
    }

    void initCoreCounters(const std::string& deviceLuid) {
        if (deviceLuid.empty() || deviceLuid.length() % 2 != 0)
            return;
        auto deviceLuidLow = deviceLuid.substr(0, 8);
        std::string luid_win;
        for (std::size_t i = 0; i < deviceLuidLow.length(); i += 2) {
            luid_win.insert(0, deviceLuidLow.substr(i, 2));
        }
        std::transform(luid_win.begin(), luid_win.end(), luid_win.begin(), std::toupper);
        std::string full3DCounterPath =
            std::string("\\GPU Engine(*_luid_*" + luid_win + "_phys*engtype_3D)\\Utilization Percentage");
        std::string fullComputeCounterPath =
            std::string("\\GPU Engine(*_luid_*" + luid_win + "_phys*engtype_Compute)\\Utilization Percentage");
        std::wstring full3DCounterPathW = ov::util::string_to_wstring(full3DCounterPath);
        std::wstring fullComputeCounterPathW = ov::util::string_to_wstring(fullComputeCounterPath);
        coreTimeCounters[luid][RENDER_ENGINE_COUNTER_INDEX] =
            addCounter(expandWildCardPath(full3DCounterPathW.c_str()));
        coreTimeCounters[luid][COMPUTE_ENGINE_COUNTER_INDEX] =
            addCounter(expandWildCardPath(fullComputeCounterPathW.c_str()));
        query.pdhCollectQueryData();
    }

    std::vector<std::wstring> expandWildCardPath(LPCWSTR WildCardPath) {
        DWORD PathListLength = 0;
        DWORD PathListLengthBufLen;
        std::vector<std::wstring> pathList;
        auto ret = query.pdhExpandWildCardPathW(NULL, WildCardPath, NULL, &PathListLength, 0);
        if (!ret) {
            return pathList;
        }
        PathListLengthBufLen = PathListLength + 100;
        PZZWSTR ExpandedPathList = (PZZWSTR)malloc(PathListLengthBufLen * sizeof(WCHAR));
        ret = query.pdhExpandWildCardPathW(NULL, WildCardPath, ExpandedPathList, &PathListLength, 0);
        if (!ret) {
            free(ExpandedPathList);
            return pathList;
        }
        for (size_t i = 0; i < PathListLength;) {
            std::wstring wpath(ExpandedPathList + i);
            if (wpath.length() > 0) {
                // std::cout << path << std::endl;
                pathList.push_back(wpath);
                i += wpath.length() + 1;
            } else {
                break;
            }
        }
        free(ExpandedPathList);
        return pathList;
    }

    std::vector<PDH_HCOUNTER> addCounter(std::vector<std::wstring> pathList) {
        std::vector<PDH_HCOUNTER> CounterList;
        for (std::wstring path : pathList) {
            PDH_HCOUNTER Counter;
            auto ret = query.pdhAddCounterW(path.c_str(), NULL, &Counter);
            if (!ret) {
                return CounterList;
            }
            ret = query.pdhSetCounterScaleFactor(Counter, -2);  // scale counter to [0, 1]
            if (!ret) {
                return CounterList;
            }
            CounterList.push_back(Counter);
        }
        return CounterList;
    }

    std::map<std::string, double> get_utilization() {
        if (luid.empty())
            return {};
        auto ts = std::chrono::system_clock::now();
        if (ts > lastTimeStamp) {
            auto delta =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastTimeStamp);
            if (delta.count() < monitor_duration) {
                std::this_thread::sleep_for(std::chrono::milliseconds(monitor_duration - delta.count()));
            }
        }
        lastTimeStamp = std::chrono::system_clock::now();
        auto ret = query.pdhCollectQueryData();
        PDH_FMT_COUNTERVALUE displayValue;
        std::map<std::string, double> utilizationMap;
        for (auto item : coreTimeCounters) {
            double utilization = 0.0;
            auto luid = item.first;
            auto coreCounters = item.second;
            for (int counterIndex = 0; counterIndex < MAX_COUNTER_INDEX; counterIndex++) {
                auto countersList = coreCounters[counterIndex];
                for (auto counter : countersList) {
                    auto status = query.pdhGetFormattedCounterValue(counter, PDH_FMT_DOUBLE, NULL, &displayValue);
                    if (status != ERROR_SUCCESS) {
                        continue;
                    }
                    utilization += displayValue.doubleValue;
                }
            }
            utilizationMap[luid] = utilization;
        }
        return utilizationMap;
    }

private:
    QueryWrapper query;
    std::map<std::string, std::vector<std::vector<PDH_HCOUNTER>>> coreTimeCounters;
    std::chrono::time_point<std::chrono::system_clock> lastTimeStamp = std::chrono::system_clock::now();
    std::string luid;
    int monitor_duration = 1000;
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
    PerformanceCounterImpl(const std::string& deviceLuid) {}

    std::map<std::string, double> get_utilization() {
        // TODO: Implement.
        return {};
    }
};

#else
namespace ov {
namespace util {
namespace monitor {
// not implemented
class GpuPerformanceCounter::PerformanceCounterImpl {
public:
    std::map<std::string, double> get_utilization() {
        return {};
    }
};
#endif
GpuPerformanceCounter::GpuPerformanceCounter(const std::string& luid) : PerformanceCounter("GPU"), deviceLuid(luid) {}
std::map<std::string, double> GpuPerformanceCounter::get_utilization() {
    if (!performance_counter)
        performance_counter = std::make_shared<PerformanceCounterImpl>(deviceLuid);
    return performance_counter->get_utilization();
}
}
}
}