// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/monitors/cpu_performance_counter.hpp"

#include <algorithm>
#include <iostream>
#include <map>

#ifdef _WIN32
#    define NOMINMAX
#    include <pdh.h>
#    include <pdhmsg.h>
#    include <windows.h>

#    include <chrono>
#    include <string>
#    include <system_error>
#    include <thread>

#    include "query_wrapper.hpp"

namespace ov {
namespace util {
namespace monitor {

class CpuPerformanceCounter::PerformanceCounterImpl {
public:
    PerformanceCounterImpl() {
        int n_cores = getNumberOfCores();
        if (n_cores == 0) {
            coreTimeCounters.resize(1);
            std::wstring fullCounterPath{L"\\Processor(_Total)\\% Processor Time"};
            auto ret = query.pdhAddCounterW(fullCounterPath.c_str(), 0, &coreTimeCounters[0]);
            if (!ret) {
                throw std::runtime_error("PdhAddCounterW() failed. Error status: " + std::to_string(ret));
            }
            ret = query.pdhSetCounterScaleFactor(coreTimeCounters[0], -2);  // scale counter to [0, 1]
            if (ret != ERROR_SUCCESS) {
                throw std::runtime_error("PdhSetCounterScaleFactor() failed. Error status: " + std::to_string(ret));
            }
        } else {
            coreTimeCounters.resize(n_cores);
            for (std::size_t i = 0; i < n_cores; ++i) {
                std::wstring fullCounterPath{L"\\Processor(" + std::to_wstring(i) + L")\\% Processor Time"};
                auto ret = query.pdhAddCounterW(fullCounterPath.c_str(), 0, &coreTimeCounters[i]);
                if (ret != ERROR_SUCCESS) {
                    throw std::runtime_error("PdhAddCounterW() failed. Error status: " + std::to_string(ret));
                }
                ret = query.pdhSetCounterScaleFactor(coreTimeCounters[i], -2);  // scale counter to [0, 1]
                if (ret != ERROR_SUCCESS) {
                    throw std::runtime_error("PdhSetCounterScaleFactor() failed. Error status: " + std::to_string(ret));
                }
            }
        }
        auto ret = query.pdhCollectQueryData();
        if (ret != ERROR_SUCCESS) {
            throw std::runtime_error("PdhCollectQueryData() failed. Error status: " + std::to_string(ret));
        }
    }

    std::map<std::string, double> get_utilization() {
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
        if (ret != ERROR_SUCCESS) {
            return {};
        }
        PDH_FMT_COUNTERVALUE displayValue;
        std::vector<double> cpuLoad(coreTimeCounters.size());
        for (std::size_t i = 0; i < coreTimeCounters.size(); ++i) {
            auto ret = query.pdhGetFormattedCounterValue(coreTimeCounters[i], PDH_FMT_DOUBLE, NULL, &displayValue);
            if (ret != ERROR_SUCCESS) {
                return {};
                if (PDH_CSTATUS_VALID_DATA != displayValue.CStatus && PDH_CSTATUS_NEW_DATA != displayValue.CStatus) {
                    throw std::runtime_error("PdhGetFormattedCounterValue() failed. Error status: " + std::to_string(ret));
                }

                cpuLoad[i] = displayValue.doubleValue * 100.0;
            }
            std::map<std::string, double> cpusUtilization;
            if (cpuLoad.size() == 1) {
                cpusUtilization["Total"] = cpuLoad.at(0);
                return cpusUtilization;
            }
            for (int index = 0; index < cpuLoad.size(); index++) {
                cpusUtilization[std::to_string(index)] = cpuLoad.at(index);
            }
            return cpusUtilization;
        }
    }

    int getNumberOfCores() {
        return 0;
    }

private:
    QueryWrapper query;
    std::vector<PDH_HCOUNTER> coreTimeCounters;
    std::chrono::time_point<std::chrono::system_clock> lastTimeStamp = std::chrono::system_clock::now();
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
class CpuPerformanceCounter::PerformanceCounterImpl {
public:
    PerformanceCounterImpl(const std::string& deviceLuid) {}

    std::map<std::string, double> get_utilization() {
        // TODO: Implement.
        return {{"Total", 0.0}};
    }
};

#else
namespace ov {
namespace util {
namespace monitor {
// not implemented
class CpuPerformanceCounter::PerformanceCounterImpl {
public:
    std::map<std::string, double> get_utilization() {
        return {{"Total", 0.0}};
    }
};
#endif
CpuPerformanceCounter::CpuPerformanceCounter(int numCores)
    : PerformanceCounter("CPU"),
      n_cores(numCores >= 0 ? numCores : 0) {}
std::map<std::string, double> CpuPerformanceCounter::get_utilization() {
    if (!performance_counter)
        performance_counter = std::make_shared<PerformanceCounterImpl>();
    if (n_cores == 0)
        return performance_counter->get_utilization();
    std::map<std::string, double> ret;
    ret["Total"] = 0.0;
    for (int i = 0; i < n_cores; i++) {
        ret[std::to_string(i)] = 0.0;
    }
    return ret;
}
}
}
}