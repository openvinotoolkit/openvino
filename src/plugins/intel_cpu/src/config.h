// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <threading/ie_istreams_executor.hpp>
#include <ie_performance_hints.hpp>
#include "utils/debug_capabilities.h"

#include <string>
#include <map>
#include <mutex>

namespace ov {
namespace intel_cpu {

struct Config {
    Config();

    enum LPTransformsMode {
        Off,
        On,
    };

    enum DenormalsOptMode {
        DO_Keep,
        DO_Off,
        DO_On,
    };

    bool collectPerfCounters = false;
    bool exclusiveAsyncRequests = false;
    bool enableDynamicBatch = false;
    std::string dumpToDot = "";
    int batchLimit = 0;
    size_t rtCacheCapacity = 5000ul;
    InferenceEngine::IStreamsExecutor::Config streamExecutorConfig;
    InferenceEngine::PerfHintsConfig  perfHintsConfig;
#if defined(__arm__) || defined(__aarch64__)
    // Currently INT8 mode is not optimized on ARM, fallback to FP32 mode.
    LPTransformsMode lpTransformsMode = LPTransformsMode::Off;
    bool enforceBF16 = false;
#else
    LPTransformsMode lpTransformsMode = LPTransformsMode::On;
    bool enforceBF16 = true;
    bool manualEnforceBF16 = false;
#endif

    std::string cache_dir{};

    DenormalsOptMode denormalsOptMode = DenormalsOptMode::DO_Keep;

    void readProperties(const std::map<std::string, std::string> &config);
    void updateProperties();
    std::map<std::string, std::string> _config;

#ifdef CPU_DEBUG_CAPS
    enum FILTER {
        BY_PORTS,
        BY_EXEC_ID,
        BY_TYPE,
        BY_NAME,
    };

    enum class FORMAT {
        BIN,
        TEXT,
    };

    std::string execGraphPath;
    std::string verbose;
    std::string blobDumpDir = "cpu_dump";
    FORMAT blobDumpFormat = FORMAT::TEXT;
    // std::hash<int> is necessary for Ubuntu-16.04 (gcc-5.4 and defect in C++11 standart)
    std::unordered_map<FILTER, std::string, std::hash<int>> blobDumpFilters;
    std::string summaryPerf = "";

    void readDebugCapsProperties();
#endif

    bool isNewApi = true;
};

}   // namespace intel_cpu
}   // namespace ov
