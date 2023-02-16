// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with common samples functionality
 * @file common.hpp
 */

#pragma once

#include <algorithm>
#include <fstream>
#include <functional>
#include <inference_engine.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

#ifndef UNUSED
    #if defined(_MSC_VER) && !defined(__clang__)
        #define UNUSED
    #else
        #define UNUSED __attribute__((unused))
    #endif
#endif
\

/**
 * @brief Get extension from filename
 * @param filename - name of the file which extension should be extracted
 * @return string with extracted file extension
 */
inline std::string fileExt(const std::string& filename) {
    auto pos = filename.rfind('.');
    if (pos == std::string::npos)
        return "";
    return filename.substr(pos + 1);
}

inline std::ostream& operator<<(std::ostream& os, const InferenceEngine::Version& version) {
    os << "\t" << version.description << " version ......... ";
    os << IE_VERSION_MAJOR << "." << IE_VERSION_MINOR << "." << IE_VERSION_PATCH;

    os << "\n\tBuild ........... ";
    os << version.buildNumber;

    return os;
}

inline std::ostream& operator<<(std::ostream& os, const InferenceEngine::Version* version) {
    if (nullptr != version) {
        os << std::endl << *version;
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const std::map<std::string, InferenceEngine::Version>& versions) {
    for (auto&& version : versions) {
        os << "\t" << version.first << std::endl;
        os << version.second << std::endl;
    }

    return os;
}

static std::vector<std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo>> perfCountersSorted(
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap) {
    using perfItem = std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo>;
    std::vector<perfItem> sorted;
    for (auto& kvp : perfMap)
        sorted.push_back(kvp);

    std::stable_sort(sorted.begin(), sorted.end(), [](const perfItem& l, const perfItem& r) {
        return l.second.execution_index < r.second.execution_index;
    });

    return sorted;
}

static UNUSED void printPerformanceCounts(const std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& performanceMap, std::ostream& stream,
                                          std::string deviceName, bool bshowHeader = true) {
    std::ios::fmtflags fmt(stream.flags());
    long long totalTime = 0;
    // Print performance counts
    if (bshowHeader) {
        stream << std::endl << "performance counts:" << std::endl << std::endl;
    }

    auto performanceMapSorted = perfCountersSorted(performanceMap);

    for (const auto& it : performanceMapSorted) {
        std::string toPrint(it.first);
        const int maxLayerName = 30;

        if (it.first.length() >= maxLayerName) {
            toPrint = it.first.substr(0, maxLayerName - 4);
            toPrint += "...";
        }

        stream << std::setw(maxLayerName) << std::left << toPrint;
        switch (it.second.status) {
        case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
            stream << std::setw(15) << std::left << "EXECUTED";
            break;
        case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
            stream << std::setw(15) << std::left << "NOT_RUN";
            break;
        case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
            stream << std::setw(15) << std::left << "OPTIMIZED_OUT";
            break;
        }
        stream << std::setw(30) << std::left << "layerType: " + std::string(it.second.layer_type) + " ";
        stream << std::setw(20) << std::left << "realTime: " + std::to_string(it.second.realTime_uSec);
        stream << std::setw(20) << std::left << "cpu: " + std::to_string(it.second.cpu_uSec);
        stream << " execType: " << it.second.exec_type << std::endl;
        if (it.second.realTime_uSec > 0) {
            totalTime += it.second.realTime_uSec;
        }
    }
    stream << std::setw(20) << std::left << "Total time: " + std::to_string(totalTime) << " microseconds" << std::endl;
    std::cout << std::endl;
    std::cout << "Full device name: " << deviceName << std::endl;
    std::cout << std::endl;
    stream.flags(fmt);
}

static UNUSED void printPerformanceCounts(InferenceEngine::InferRequest request, std::ostream& stream, std::string deviceName, bool bshowHeader = true) {
    auto performanceMap = request.GetPerformanceCounts();
    printPerformanceCounts(performanceMap, stream, deviceName, bshowHeader);
}

inline std::string getFullDeviceName(std::map<std::string, std::string>& devicesMap, std::string device) {
    std::map<std::string, std::string>::iterator it = devicesMap.find(device);
    if (it != devicesMap.end()) {
        return it->second;
    } else {
        return "";
    }
}

inline std::string getFullDeviceName(InferenceEngine::Core& ie, std::string device) {
    InferenceEngine::Parameter p;
    try {
        p = ie.GetMetric(device, METRIC_KEY(FULL_DEVICE_NAME));
        return p.as<std::string>();
    } catch (InferenceEngine::Exception&) {
        return "";
    }
}

inline void showAvailableDevices() {
    InferenceEngine::Core ie;
    std::vector<std::string> devices = ie.GetAvailableDevices();

    std::cout << std::endl;
    std::cout << "Available target devices:";
    for (const auto& device : devices) {
        std::cout << "  " << device;
    }
    std::cout << std::endl;
}
