// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/monitors/cpu_performance_counter.hpp"

#include <algorithm>
#include <iostream>

#ifdef _WIN32
#    define NOMINMAX
#    include <PdhMsg.h>
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
        PDH_STATUS status;
        int nCores = getNumberOfCores();
        if (nCores == 0) {
            coreTimeCounters.resize(1);
            std::wstring fullCounterPath{L"\\Processor(_Total)\\% Processor Time"};
            status = PdhAddCounterW(query, fullCounterPath.c_str(), 0, &coreTimeCounters[0]);
            if (ERROR_SUCCESS != status) {
                throw std::system_error(status, std::system_category(), "PdhAddCounterW() failed");
            }
            status = PdhSetCounterScaleFactor(coreTimeCounters[0], -2);  // scale counter to [0, 1]
            if (ERROR_SUCCESS != status) {
                throw std::system_error(status, std::system_category(), "PdhSetCounterScaleFactor() failed");
            }
        } else {
            coreTimeCounters.resize(nCores);
            for (std::size_t i = 0; i < nCores; ++i) {
                std::wstring fullCounterPath{L"\\Processor(" + std::to_wstring(i) + L")\\% Processor Time"};
                status = PdhAddCounterW(query, fullCounterPath.c_str(), 0, &coreTimeCounters[i]);
                if (ERROR_SUCCESS != status) {
                    throw std::system_error(status, std::system_category(), "PdhAddCounterW() failed");
                }
                status = PdhSetCounterScaleFactor(coreTimeCounters[i], -2);  // scale counter to [0, 1]
                if (ERROR_SUCCESS != status) {
                    throw std::system_error(status, std::system_category(), "PdhSetCounterScaleFactor() failed");
                }
            }
        }
        status = PdhCollectQueryData(query);
        if (ERROR_SUCCESS != status) {
            throw std::system_error(status, std::system_category(), "PdhCollectQueryData() failed");
        }
    }

    std::vector<double> getCpuLoad() {
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
        if (ERROR_SUCCESS != status) {
            throw std::system_error(status, std::system_category(), "PdhCollectQueryData() failed");
        }
        PDH_FMT_COUNTERVALUE displayValue;
        std::vector<double> cpuLoad(coreTimeCounters.size());
        for (std::size_t i = 0; i < coreTimeCounters.size(); ++i) {
            status = PdhGetFormattedCounterValue(coreTimeCounters[i], PDH_FMT_DOUBLE, NULL, &displayValue);
            switch (status) {
            case ERROR_SUCCESS:
                break;
            // PdhGetFormattedCounterValue() can sometimes return PDH_CALC_NEGATIVE_DENOMINATOR for some reason
            case PDH_CALC_NEGATIVE_DENOMINATOR:
                return {};
            default:
                throw std::system_error(status, std::system_category(), "PdhGetFormattedCounterValue() failed");
            }
            if (PDH_CSTATUS_VALID_DATA != displayValue.CStatus && PDH_CSTATUS_NEW_DATA != displayValue.CStatus) {
                throw std::runtime_error("Error in counter data");
            }

            cpuLoad[i] = displayValue.doubleValue;
        }
        return cpuLoad;
    }

    int getNumberOfCores() {
        return 0;
    }

private:
    QueryWrapper query;
    std::vector<PDH_HCOUNTER> coreTimeCounters;
    std::chrono::time_point<std::chrono::system_clock> lastTimeStamp = std::chrono::system_clock::now();
};

#elif __linux__
#    include <unistd.h>

#    include <chrono>
#    include <fstream>
#    include <regex>
#    include <utility>

namespace {
const long clockTicks = sysconf(_SC_CLK_TCK);

const std::size_t numCores = sysconf(_SC_NPROCESSORS_CONF);

std::vector<unsigned long> getIdleCpuStat() {
    std::vector<unsigned long> idleCpuStat(numCores);
    std::ifstream procStat("/proc/stat");
    std::string line;
    std::smatch match;
    std::regex coreJiffies("^cpu(\\d+)\\s+"
                           "(\\d+)\\s+"
                           "(\\d+)\\s+"
                           "(\\d+)\\s+"
                           "(\\d+)\\s+"  // idle
                           "(\\d+)");    // iowait

    while (std::getline(procStat, line)) {
        if (std::regex_search(line, match, coreJiffies)) {
            // it doesn't handle overflow of sum and overflows of /proc/stat values
            unsigned long idleInfo = stoul(match[5]) + stoul(match[6]), coreId = stoul(match[1]);
            if (numCores <= coreId) {
                throw std::runtime_error("The number of cores has changed");
            }
            idleCpuStat[coreId] = idleInfo;
        }
    }
    return idleCpuStat;
}
}  // namespace

namespace ov {
namespace util {
namespace monitor {
class CpuPerformanceCounter::PerformanceCounterImpl {
public:
    PerformanceCounterImpl() : prevIdleCpuStat{getIdleCpuStat()}, prevTimePoint{std::chrono::steady_clock::now()} {}

    std::vector<double> getLoad() {
        std::vector<unsigned long> idleCpuStat = getIdleCpuStat();
        auto timePoint = std::chrono::steady_clock::now();
        // don't update data too frequently which may result in negative values for cpuLoad.
        // It may happen when collectData() is called just after setHistorySize().
        if (timePoint - prevTimePoint > std::chrono::milliseconds{300}) {
            std::vector<double> cpuLoad(numCores);
            for (std::size_t i = 0; i < idleCpuStat.size(); ++i) {
                double idleDiff = idleCpuStat[i] - prevIdleCpuStat[i];
                typedef std::chrono::duration<double, std::chrono::seconds::period> Sec;
                cpuLoad[i] =
                    1.0 - idleDiff / clockTicks / std::chrono::duration_cast<Sec>(timePoint - prevTimePoint).count();
            }
            prevIdleCpuStat = std::move(idleCpuStat);
            prevTimePoint = timePoint;
            return cpuLoad;
        }
        return {};
    }

private:
    std::vector<unsigned long> prevIdleCpuStat;
    std::chrono::steady_clock::time_point prevTimePoint;
};

#else
// not implemented
namespace {
const std::size_t nCores{0};
}

class CpuMonitor::PerformanceCounterImpl {
public:
    std::vector<double> getCpuLoad() {
        return {};
    };
};
#endif
CpuPerformanceCounter::CpuPerformanceCounter(int numCores)
    : PerformanceCounter("CPU"),
      nCores(numCores >= 0 ? numCores : 0) {}
CpuPerformanceCounter::~CpuPerformanceCounter() {
    delete performanceCounter;
}
std::vector<double> CpuPerformanceCounter::getLoad() {
    if (!performanceCounter)
        performanceCounter = new PerformanceCounterImpl();
    return performanceCounter->getLoad();
}
}
}
}