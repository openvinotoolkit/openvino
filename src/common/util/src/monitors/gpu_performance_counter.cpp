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
        std::string luid_win;
        for (std::size_t i = 0; i < deviceLuid.length(); i += 2) {
            luid_win.insert(0, deviceLuid.substr(i, 2));
        }
        luid_win = luid_win.length() > 8 ? luid_win.substr(0, 8) : luid_win;
        std::transform(luid_win.begin(), luid_win.end(), luid_win.begin(), std::toupper);
        std::string full3DCounterPath =
            std::string("\\GPU Engine(*_luid_*" + luid_win + "_phys*engtype_3D)\\Utilization Percentage");
        std::string fullComputeCounterPath =
            std::string("\\GPU Engine(*_luid_*" + luid_win + "_phys*engtype_Compute)\\Utilization Percentage");
        coreTimeCounters[luid][RENDER_ENGINE_COUNTER_INDEX] =
            addCounter(query, expandWildCardPath(full3DCounterPath.c_str()));
        coreTimeCounters[luid][COMPUTE_ENGINE_COUNTER_INDEX] =
            addCounter(query, expandWildCardPath(fullComputeCounterPath.c_str()));
        PdhCollectQueryData(query);
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

    std::map<std::string, double> getUtilization() {
        if (luid.empty())
            return {};
        PDH_STATUS status;
        auto ts = std::chrono::system_clock::now();
        if (ts > lastTimeStamp) {
            auto delta =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastTimeStamp);
            if (delta.count() < monitor_duration) {
                std::this_thread::sleep_for(std::chrono::milliseconds(monitor_duration - delta.count()));
            }
        }
        lastTimeStamp = std::chrono::system_clock::now();
        status = PdhCollectQueryData(query);
        PDH_FMT_COUNTERVALUE displayValue;
        double utilization = 0.0;
        for (auto item : coreTimeCounters) {
            auto coreCounters = item.second;
            for (int counterIndex = 0; counterIndex < MAX_COUNTER_INDEX; counterIndex++) {
                auto countersList = coreCounters[counterIndex];
                for (auto counter : countersList) {
                    status = PdhGetFormattedCounterValue(counter, PDH_FMT_DOUBLE, NULL, &displayValue);
                    if (status != ERROR_SUCCESS) {
                        continue;
                    }
                    utilization += displayValue.doubleValue;
                }
            }
        }
        return {{luid, utilization}};
    }

private:
    QueryWrapper query;
    std::map<std::string, std::vector<std::vector<PDH_HCOUNTER>>> coreTimeCounters;
    std::chrono::time_point<std::chrono::system_clock> lastTimeStamp = std::chrono::system_clock::now();
    std::string luid;
    int monitor_duration = 500;
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

    std::map<std::string, double> getUtilization() {
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
    std::map<string, double> getUtilization() {
        return {};
    }
};
#endif
GpuPerformanceCounter::GpuPerformanceCounter(const std::string& luid) : PerformanceCounter("GPU"), deviceLuid(luid) {}
std::map<std::string, double> GpuPerformanceCounter::getUtilization() {
    if (!performance_counter)
        performance_counter = std::make_shared<PerformanceCounterImpl>(deviceLuid);
    return performance_counter->getUtilization();
}
}
}
}